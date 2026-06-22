import math
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix


def build_sparse_token_feature_matrix_continuous(
    df: pd.DataFrame,
    feature_col: str = "feature_ids",
    activation_col: str = "feature_acts",
):
    rows = []
    cols = []
    data = []

    feature_to_col = {}
    col_to_feature = []

    for token_idx, (feats, acts) in tqdm(enumerate(zip(df[feature_col], df[activation_col])), total=len(df), desc="Building continuous feature matrix", leave=False):
        if not isinstance(feats, (list, tuple, set, np.ndarray)):
            continue

        seen_features = {}
        for f, act in zip(feats, acts):
            if pd.isna(f):
                continue
            if f not in seen_features or act > seen_features[f]:
                seen_features[f] = act

        for f, act in seen_features.items():
            if f not in feature_to_col:
                feature_to_col[f] = len(col_to_feature)
                col_to_feature.append(f)

            rows.append(token_idx)
            cols.append(feature_to_col[f])
            data.append(act)

    X = sparse.csr_matrix(
        (np.array(data, dtype=np.float32), (rows, cols)),
        shape=(len(df), len(col_to_feature)),
        dtype=np.float32,
    )

    return X, np.array(col_to_feature, dtype=object)

def build_sparse_token_feature_matrix(
    df: pd.DataFrame,
    feature_col: str = "feature_ids",
):
    rows = []
    cols = []

    feature_to_col = {}
    col_to_feature = []

    for token_idx, feats in enumerate(df[feature_col]):
        if not isinstance(feats, (list, tuple, set, np.ndarray)):
            continue

        for f in set(feats):
            if pd.isna(f):
                continue
            if f not in feature_to_col:
                feature_to_col[f] = len(col_to_feature)
                col_to_feature.append(f)

            rows.append(token_idx)
            cols.append(feature_to_col[f])

    data = np.ones(len(rows), dtype=np.float32)

    X = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(df), len(col_to_feature)),
        dtype=np.float32,
    )

    return X, np.array(col_to_feature, dtype=object)


def build_sparse_token_concept_matrix(
    df: pd.DataFrame,
    concepts: list[str],
    concepts_col: str = "concepts",
):
    concept_to_col = {c: i for i, c in enumerate(concepts)}

    rows = []
    cols = []

    for token_idx, cs in enumerate(df[concepts_col]):
        if not isinstance(cs, (list, tuple, set, np.ndarray)):
            continue

        for c in set(cs):
            if c in concept_to_col:
                rows.append(token_idx)
                cols.append(concept_to_col[c])

    data = np.ones(len(rows), dtype=np.float32)

    C = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(df), len(concepts)),
        dtype=np.float32,
    )

    return C


def compute_all_concept_feature_metrics(
    df: pd.DataFrame,
    concepts: list[str],
    X_boolean: csc_matrix,
    C_sparse: csc_matrix,
    feature_ids_mapping: np.ndarray,
    feature_col: str = "feature_ids",
    concepts_col: str = "concepts",
    min_joint_count: int = 3,
    min_feature_count: int = 5,
    compute_auc: bool = False,
) -> pd.DataFrame:
    """
    Calcule les métriques feature/concept pour tous les concepts en une seule passe.

    Retourne une ligne par paire (concept, feature_id).
    """

    required_columns = {feature_col, concepts_col}
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    if df.empty:
        return pd.DataFrame()

    n_tokens = len(df)

    # count_feature[f]
    count_feature = np.asarray(X_boolean.sum(axis=0)).ravel()

    # count_concept[c]
    count_concept = np.asarray(C_sparse.sum(axis=0)).ravel()

    # joint[c, f] = nombre de tokens où concept c et feature f coexistent
    joint = C_sparse.T @ X_boolean
    joint = joint.tocoo()

    concept_idx = joint.row
    feature_idx = joint.col
    count_joint = joint.data.astype(np.int64)

    valid = (
        (count_joint >= min_joint_count)
        & (count_feature[feature_idx] >= min_feature_count)
        & (count_concept[concept_idx] > 0)
    )

    concept_idx = concept_idx[valid]
    feature_idx = feature_idx[valid]
    count_joint = count_joint[valid]

    if len(count_joint) == 0:
        return pd.DataFrame()

    cf = count_feature[feature_idx]
    cc = count_concept[concept_idx]

    p_feature = cf / n_tokens
    p_concept = cc / n_tokens
    p_joint = count_joint / n_tokens

    pmi = np.log2(p_joint / (p_feature * p_concept))
    npmi = pmi / (-np.log2(p_joint))

    precision = count_joint / cf
    recall = count_joint / cc

    lift = precision / (cc / n_tokens)
    specificity = (cf - count_joint) / (n_tokens - cc)

    f1 = np.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )

    result = pd.DataFrame({
        "concept": np.array(concepts, dtype=object)[concept_idx],
        "feature_id": feature_ids_mapping[feature_idx],
        "count_feature": cf.astype(int),
        "count_concept": cc.astype(int),
        "count_joint": count_joint.astype(int),
        "p_feature": p_feature,
        "p_concept": p_concept,
        "p_joint": p_joint,
        "pmi": pmi,
        "npmi": npmi,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lift": lift,
        "specificity": specificity,
    })

    if compute_auc:
        auprc_values = []
        auroc_values = []

        X_csc = X_boolean.tocsc()
        C_csc = C_sparse.tocsc()

        for c, f in tqdm(zip(concept_idx, feature_idx), total=len(concept_idx), desc="Computing AUPRC/AUROC", leave=False):
            y_true = np.zeros(n_tokens, dtype=np.int8)
            y_score = np.zeros(n_tokens, dtype=np.int8)

            y_true[C_csc[:, c].indices] = 1
            y_score[X_csc[:, f].indices] = 1

            auprc_values.append(average_precision_score(y_true, y_score))

            if y_true.min() == y_true.max():
                auroc_values.append(np.nan)
            else:
                auroc_values.append(roc_auc_score(y_true, y_score))

        result["auprc"] = auprc_values
        result["auroc"] = auroc_values

    return result.sort_values(
        [
            "concept",
            "pmi",
            "precision",
            "recall",
            "f1",
            "count_joint",
            "feature_id",
        ],
        ascending=[True, False, False, False, False, False, True],
    ).reset_index(drop=True)


def compute_all_features_ast_purity(
    df: pd.DataFrame,
    ast_type_col: str = "ast_type",
    feature_id_col: str = "feature_ids",
    feature_act_col: str = "feature_acts"
) -> pd.DataFrame:
    """
    Calcule le score de pureté AST (1 - Entropie Normalisée) pour TOUTES les features,
    ainsi que leur volume d'activation et leur dispersion syntaxique.
    
    Retourne un DataFrame contenant une ligne par feature_id.
    """
    # 1. Validation des colonnes nécessaires
    required = {ast_type_col, feature_id_col, feature_act_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Colonnes manquantes dans le DataFrame : {missing}")
        
    if df.empty:
        return pd.DataFrame(columns=[
            "feature_id", "ast_purity", "total_activation", 
            "count_activations", "count_distinct_ast_types"
        ])

    # 2. Calcul du nombre total de types AST uniques dans tout le dataset (pour la normalisation)
    n_distinct_ast_types = df[ast_type_col].nunique()
    if n_distinct_ast_types <= 1:
        distinct_features = set([f for sublist in df[feature_id_col] for f in sublist if pd.notna(f)])
        return pd.DataFrame({
            "feature_id": list(distinct_features),
            "ast_purity": 1.0,
            "total_activation": np.nan,
            "count_activations": np.nan,
            "count_distinct_ast_types": int(n_distinct_ast_types)
        })

    max_entropy = np.log2(n_distinct_ast_types)

    # 3. Mise à plat (Explode) des colonnes de listes alignées
    exploded_df = df[[ast_type_col, feature_id_col, feature_act_col]].copy()
    exploded_df = exploded_df.explode([feature_id_col, feature_act_col])

    # 4. Nettoyage et filtrage des activations strictes (> 0)
    exploded_df = exploded_df.dropna(subset=[feature_id_col, feature_act_col])
    exploded_df[feature_act_col] = exploded_df[feature_act_col].astype(np.float32)
    exploded_df = exploded_df[exploded_df[feature_act_col] > 0]

    if exploded_df.empty:
        return pd.DataFrame(columns=[
            "feature_id", "ast_purity", "total_activation", 
            "count_activations", "count_distinct_ast_types"
        ])

    # =========================================================================
    # ÉTAPE CLÉ : On extrait le nombre total d'activations par feature (en volume)
    # AVANT de grouper par type AST, pour ne pas perdre la granularité du token.
    # =========================================================================
    feature_counts = exploded_df.groupby(feature_id_col).size().rename("count_activations")

    # 5. Agrégation par couple (feature_id, ast_type) pour la distribution d'intensité
    grouped = exploded_df.groupby([feature_id_col, ast_type_col])[feature_act_col].sum().reset_index()

    # 6. Calcul de la distribution de probabilité P(ast_type | feature_id)
    feature_totals = grouped.groupby(feature_id_col)[feature_act_col].transform("sum")
    grouped["p"] = grouped[feature_act_col] / feature_totals

    # 7. Calcul de la contribution à l'entropie : -p * log2(p)
    grouped["entropy_contrib"] = -grouped["p"] * np.log2(grouped["p"])

    # 8. Somme de l'entropie et extraction du nombre de types AST uniques par feature
    # (Puisque 'grouped' contient une ligne par couple unique existant, .size() donne le nombre d'AST uniques)
    final_features = grouped.groupby(feature_id_col).agg(
        total_entropy=("entropy_contrib", "sum"),
        total_activation=(feature_act_col, "sum"),
        count_distinct_ast_types=(ast_type_col, "size")
    ).reset_index()

    # 9. Normalisation et calcul du score de pureté final (1 - H_norm)
    final_features["ast_purity"] = 1.0 - (final_features["total_entropy"] / max_entropy)

    # 10. Jointure en O(1) avec les volumes d'activations calculés à l'étape 4
    final_features = final_features.merge(feature_counts, on=feature_id_col, how="left")

    # Nettoyage et tri
    final_features = final_features.rename(columns={feature_id_col: "feature_id"})
    
    # Réorganisation des colonnes pour la lisibilité
    column_order = [
        "feature_id", 
        "ast_purity", 
        "total_activation", 
        "count_activations", 
        "count_distinct_ast_types"
    ]
    final_features = final_features[column_order].sort_values(by="ast_purity", ascending=False).reset_index(drop=True)

    return final_features

def group_by_individual_concepts(
    df: pd.DataFrame,
    ast_type_col: str = "ast_type",
    path_col: str = "path",
    token_index_col: str = "token_index",
    feature_id_col: str = "feature_ids",
    feature_act_col: str = "feature_activations",
    debug_limit: int = 0
) -> dict[str, pd.DataFrame]:
    """Extrait tous les concepts uniques du DataFrame et renvoie un dictionnaire

    où chaque clé est un concept et chaque valeur est le DataFrame fusionné par
    blocs contigus de ce concept.
    """
    if df.empty:
        return {}

    # 1. Lister tous les concepts uniques existants dans le dataset
    all_concepts = set()
    for sublist in tqdm(df[ast_type_col], desc="Listing concepts", leave=False):
        if isinstance(sublist, (list, tuple, set, np.ndarray)):
            for c in sublist:
                if pd.notna(c):
                    all_concepts.add(c)

    # Fonction interne d'aggrégation des features au sein d'un bloc fusionné
    def aggregate_features(series_ids, series_acts):
        merged_feats = {}
        for ids, acts in zip(series_ids, series_acts):
            if not isinstance(ids, (list, np.ndarray)):
                continue
            for f_id, act in zip(ids, acts):
                merged_feats[f_id] = max(merged_feats.get(f_id, 0.0), act)
        return list(merged_feats.keys()), list(merged_feats.values())

    results = {}
    i = 0
    # 2. Pour chaque concept, générer son sous-dataset segmenté
    for concept in tqdm(all_concepts, desc="Processing concepts", total=debug_limit if debug_limit > 0 else len(all_concepts), leave=False):
        # Masque booléen : le concept est-il présent dans la liste de la ligne ?
        has_concept = df[ast_type_col].apply(
            lambda x: concept in x
            if isinstance(x, (list, tuple, set, np.ndarray))
            else False
        )

        # On isole uniquement les lignes contenant le concept
        concept_df = df[has_concept].copy()

        if concept_df.empty:
            continue

        # 3. Calcul de la contiguïté sur le sous-ensemble filtré
        # Attention : si l'index d'origine n'est pas contigu (ex: saut de 1 à 3),
        # ou si on change de fichier, on doit casser le bloc.
        index_jump = concept_df[token_index_col] != concept_df[
            token_index_col
        ].shift(1) + 1
        path_changed = concept_df[path_col] != concept_df[path_col].shift(1)

        # ID de bloc unique pour ce concept
        block_id = (index_jump | path_changed).cumsum()

        # 4. Aggrégation par bloc contigu
        grouped_df = (
            concept_df.groupby(block_id)
            .agg(
                token_index=(token_index_col, list),
                path=(path_col, "first"),
                _raw_ids=(feature_id_col, list),
                _raw_acts=(feature_act_col, list),
            )
            .reset_index(drop=True)
        )

        # 5. Fusion vectorisée des listes parallèles de features/activations
        merged_features = [
            aggregate_features(ids, acts)
            for ids, acts in zip(grouped_df["_raw_ids"], grouped_df["_raw_acts"])
        ]

        grouped_df["feature_ids"] = [item[0] for item in merged_features]
        grouped_df["feature_activations"] = [item[1] for item in merged_features]

        # Nettoyage final des colonnes de la Test Suite
        final_cols = ["feature_ids", "feature_activations", "token_index", "path"]
        results[concept] = grouped_df[final_cols]

        if debug_limit > 0 and i + 1 >= debug_limit:
            break

        i += 1

    return results

def group_by_individual_concepts_no_aggregation(
    df: pd.DataFrame,
    concepts_col: str = "concepts",
    path_col: str = "path",
    token_index_col: str = "token_index",
) -> dict[str, list[np.ndarray]]:
    """Version ultra-rapide sans boucle concept par concept.
    
    Retourne un dictionnaire où chaque valeur est une liste d'arrays NumPy 
    contenant les indices globaux des tokens de chaque bloc.
    """
    if df.empty:
        return {}

    df_working = df[[concepts_col, path_col, token_index_col]].copy()
    df_working["global_row_idx"] = np.arange(len(df_working))

    # 2. Mise à plat globale de tous les concepts simultanément
    df_exploded = df_working.explode(concepts_col).dropna(subset=[concepts_col])

    # 
    df_exploded = df_exploded.sort_values([concepts_col, path_col, token_index_col])

    concept_changed = df_exploded[concepts_col] != df_exploded[concepts_col].shift(1)
    path_changed = df_exploded[path_col] != df_exploded[path_col].shift(1)
    index_jump = df_exploded[token_index_col] != df_exploded[token_index_col].shift(1) + 1

    block_id = (concept_changed | path_changed | index_jump).cumsum()

    # 5. Agrégation native : on collecte les indices de ligne globaux de la matrice creuse
    grouped = df_exploded.groupby(block_id).agg(
        concept=(concepts_col, "first"),
        token_indices=("global_row_idx", list)
    )

    # 6. Dispatching dans le dictionnaire final sous forme d'arrays NumPy
    results = {}
    for c_name, group_df in grouped.groupby("concept"):
        results[c_name] = [np.array(idx, dtype=np.int64) for idx in group_df["token_indices"]]

    return results

def compute_tcf_metrics(
    df_metrics: pd.DataFrame,
    df_blocks_dict: dict[str, list[np.ndarray]], # Attend le nouveau dictionnaire d'indices
    X_continuous: csc_matrix,
    C_sparse: csc_matrix,
    concepts_list: list[str],
    feature_ids_mapping: np.ndarray,
    n_bins: int = 20
) -> pd.DataFrame:
    """
    Calcule la TCF en extrayant à la volée le maximum d'activation par bloc
    via un masque d'indexation direct en mémoire.
    """
    if df_metrics.empty:
        return df_metrics

    concept_to_idx = {name: i for i, name in enumerate(concepts_list)}
    feature_to_idx = {f_id: i for i, f_id in enumerate(feature_ids_mapping)}

    tcf_values, precision_at_opt_tau, recall_at_opt_tau, span_recall_at_opt_tau, optimal_thresholds = [], [], [], [], []

    X_csc = X_continuous.tocsc()
    C_csc = C_sparse.tocsc()
    
    pbar = tqdm(df_metrics.iterrows(), total=len(df_metrics), desc="Computing TCF Metric", unit="pair", leave=False)
    for _, row in pbar:
        c_name = row["concept"]
        f_id = row["feature_id"]
        pbar.set_postfix({"concept": c_name, "feature_id": f_id})

        if c_name not in concept_to_idx or f_id not in feature_to_idx or c_name not in df_blocks_dict:
            for l in [tcf_values, precision_at_opt_tau, recall_at_opt_tau, span_recall_at_opt_tau, optimal_thresholds]:
                l.append(0.0)
            continue

        c_idx = concept_to_idx[c_name]
        f_idx = feature_to_idx[f_id]

        y_true = np.zeros(X_csc.shape[0], dtype=np.int8)
        y_true[C_csc[:, c_idx].indices] = 1
        count_concept_tokens = y_true.sum()

        feat_col = X_csc[:, f_idx]
        activations = feat_col.data
        token_indices = feat_col.indices

        span_groups = df_blocks_dict[c_name]
        total_span_groups = len(span_groups)

        if len(activations) == 0 or count_concept_tokens == 0 or total_span_groups == 0:
            for l in [tcf_values, precision_at_opt_tau, recall_at_opt_tau, span_recall_at_opt_tau, optimal_thresholds]:
                l.append(0.0)
            continue

        token_to_act = np.zeros(X_csc.shape[0], dtype=np.float32)
        token_to_act[token_indices] = activations

        max_acts_per_block = np.array([np.max(token_to_act[group]) for group in span_groups], dtype=np.float32)

        # 4. Seuils candidats
        thresholds = np.linspace(0.0, activations.max(), num=n_bins)

        best_tcf, best_precision, best_recall, best_span_recall, best_threshold = 0.0, 0.0, 0.0, 0.0, 0.0
        is_concept_token = y_true[token_indices]

        # 5. Balayage des seuils
        for tau in thresholds:
            true_positives = np.sum((activations > tau) & is_concept_token)
            total_predicted_positive = np.sum(activations > tau)

            if true_positives == 0:
                continue

            precision = true_positives / total_predicted_positive
            recall = true_positives / count_concept_tokens

            # Calcul du Span-Recall vectoriel ultra-rapide
            successful_spans = np.sum(max_acts_per_block > tau)
            span_recall = successful_spans / total_span_groups

            if (span_recall + precision) > 0:
                tcf = 2 * (span_recall * precision) / (span_recall + precision)
            else:
                tcf = 0.0

            if tcf > best_tcf:
                best_tcf, best_precision, best_recall, best_span_recall, best_threshold = tcf, precision, recall, span_recall, tau
            elif precision < best_precision and span_recall < best_span_recall:
                break # Early stopping préservé

        tcf_values.append(best_tcf)
        precision_at_opt_tau.append(best_precision)
        recall_at_opt_tau.append(best_recall)
        span_recall_at_opt_tau.append(best_span_recall)
        optimal_thresholds.append(best_threshold)

    df_metrics["tcf"] = tcf_values
    df_metrics["precision"] = precision_at_opt_tau
    df_metrics["recall"] = recall_at_opt_tau
    df_metrics["span_recall"] = span_recall_at_opt_tau
    df_metrics["optimal_threshold"] = optimal_thresholds

    return df_metrics