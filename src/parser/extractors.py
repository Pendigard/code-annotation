import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol


def find_identifier_span(line: str, identifier: str) -> tuple[int, int] | None:
    pattern = rf"\b{re.escape(identifier)}\b"

    match = re.search(pattern, line)
    return None if match is None else match.span()

# -------------------
# | BASE EXTRACTORS |
# -------------------

@dataclass
class Neo4jConceptExtractor(Protocol):
    name: str
    query: str
    
    def get_annotations(
        self,
        neo4j_driver: Any,
        source_dir: Path,
    ) -> List[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = []
        unmatched = 0

        with neo4j_driver.session() as session:
            for record in session.run(self.query):
                annotation = self._record_to_annotation(record, source_dir)

                if annotation is None:
                    unmatched += 1
                    continue

                annotations.append(annotation)

        print(f"Unmatched variables: {unmatched}")
        return annotations
    
    def _record_to_annotation(
        self,
        record: Any,
        source_dir: Path,
    ) -> Dict[str, Any] | None:
        ...

@dataclass
class DeclarationExtractor(Neo4jConceptExtractor, Protocol):
    """Extract variable and parameter declarations."""

    def _record_to_annotation(
        self,
        record: Any,
        source_dir: Path,
    ) -> Dict[str, Any] | None:
        file_path = source_dir / record["filename"]
        line_number = record["decl"]["LINE_NUMBER"]

        if line_number is None or not file_path.exists():
            return None

        lines = file_path.read_text().splitlines()

        if line_number < 1 or line_number > len(lines):
            return None

        code = lines[line_number - 1]
        if code is None or record["decl"]["NAME"] is None:
            return None

        line_span = find_identifier_span(code, record["decl"]["NAME"])



        if line_span is None:
            return None

        offset = sum(len(lines[i]) + 1 for i in range(line_number - 1))
        span = (offset + line_span[0], offset + line_span[1])
        print(record["decl"]["NAME"], file_path.read_text()[span[0]:span[1]])
        return {
            "concept": self.name,
            "path": str(file_path),
            "variable_name": record["decl"]["NAME"],
            "line": line_number,
            "column": record["decl"]["COLUMN_NUMBER"],
            "code": code.strip(),
            "span_start": span[0],
            "span_end": span[1]
        }

# ----------------------
# | CONCEPT EXTRACTORS |
# ----------------------
@dataclass
class FixedValue(DeclarationExtractor):
    """Extract variables and parameters that are never reassigned after declaration."""

    name: str = "fixed_value"
    query: str = """
MATCH (decl:LOCAL)<-[r:REF]-(assign:IDENTIFIER)<-[arg:ARGUMENT]-(c:CALL)
WHERE (decl.LINE_NUMBER = assign.LINE_NUMBER AND decl.COLUMN_NUMBER = assign.COLUMN_NUMBER) 
AND (c.NAME = "<operator>.assignment" AND "LOCAL" IN labels(decl))
AND assign.ARGUMENT_INDEX = 1
AND NOT EXISTS { 
(decl)<-[r2:REF]-(assign2:IDENTIFIER)<-[arg2:ARGUMENT]-(c2:CALL)
  WHERE c2.NAME IN [ // All the writing operator
    "<operator>.assignment",
    "<operator>.assignmentPlus",
    "<operator>.assignmentMinus",
    "<operator>.assignmentMultiplication",
    "<operator>.assignmentDivision",
    "<operator>.postIncrement",
    "<operator>.postDecrement",
    "<operator>.preIncrement",
    "<operator>.preDecrement"
  ] AND assign2.ARGUMENT_INDEX = 1 // Left side of the assignment
  AND (decl.LINE_NUMBER <> assign2.LINE_NUMBER OR decl.COLUMN_NUMBER <> assign2.COLUMN_NUMBER) // Not counting the declaration if it is an assignment
}
MATCH (decl)<-[:AST*]-(m:METHOD)
RETURN DISTINCT decl, m.FILENAME AS filename

UNION

MATCH (decl:METHOD_PARAMETER_IN)
WHERE NOT EXISTS { 
(decl)<-[r2:REF]-(assign2:IDENTIFIER)<-[arg2:ARGUMENT]-(c2:CALL)
  WHERE c2.NAME IN [ // All the writing operator
    "<operator>.assignment",
    "<operator>.assignmentPlus",
    "<operator>.assignmentMinus",
    "<operator>.assignmentMultiplication",
    "<operator>.assignmentDivision",
    "<operator>.postIncrement",
    "<operator>.postDecrement",
    "<operator>.preIncrement",
    "<operator>.preDecrement"
  ] AND assign2.ARGUMENT_INDEX = 1 // Left side of the assignment
  AND (decl.LINE_NUMBER <> assign2.LINE_NUMBER OR decl.COLUMN_NUMBER <> assign2.COLUMN_NUMBER) // Not counting the declaration if it is an assignment
}
MATCH (decl)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
RETURN DISTINCT decl, m.FILENAME AS filename
"""

@dataclass
class FutureMutatedVariable(DeclarationExtractor):
    """Extract variables and parameters, then label them as stable or future-mutated."""

    name: str = "future_mutated_variable"

    query: str = """
// Find for all declaration, count all the identifier that are reassigned
MATCH (decl:LOCAL|METHOD_PARAMETER_IN)<-[r:REF]-(assign:IDENTIFIER)<-[arg:ARGUMENT]-(c:CALL)
WHERE c.NAME IN [ // All the writing operator
  "<operator>.assignment",
  "<operator>.assignmentPlus",
  "<operator>.assignmentMinus",
  "<operator>.assignmentMultiplication",
  "<operator>.assignmentDivision",
  "<operator>.postIncrement",
  "<operator>.postDecrement",
  "<operator>.preIncrement",
  "<operator>.preDecrement"
] AND assign.ARGUMENT_INDEX = 1 // Left side of the assignment
AND (decl.LINE_NUMBER <> assign.LINE_NUMBER OR decl.COLUMN_NUMBER <> assign.COLUMN_NUMBER) // Not counting the declaration if it is an assignment
WITH decl, count(assign) AS num_writes
MATCH (decl)<-[:AST*]-(m:METHOD)
RETURN DISTINCT decl, m.FILENAME as filename, num_writes
"""

    def get_annotations(
        self,
        neo4j_driver: Any,
        source_dir: Path,
    ) -> List[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = []
        unmatched = 0

        with neo4j_driver.session() as session:
            for record in session.run(self.query):
                annotation = self._record_to_annotation(record, source_dir)

                if annotation is None:
                    unmatched += 1
                    continue

                annotation["num_writes"] = record["num_writes"]

                annotations.append(annotation)

        print(f"Unmatched variables: {unmatched}")
        return annotations

@dataclass
class Gatherer(DeclarationExtractor):
    """Extract variables that are used to accumulate state across method calls."""

    name: str = "gatherer"

    query: str = """
MATCH (decl:LOCAL)<-[:REF]-(assign:IDENTIFIER)<-[:ARGUMENT]-(c_assign:CALL)
// Match a declared identifier
      , (cs:CONTROL_STRUCTURE)<-[:AST]-(n)-[:AST]->(decl)
// That is declared before a loop
      , (decl)<-[:REF]-(update:IDENTIFIER)<-[:ARGUMENT]-(c_updt:CALL)
// This identifier is updated
      , (c_updt)<-[:AST*]-(cs)
// The update is inside the loop

WHERE c_assign.NAME = "<operator>.assignment"
  AND assign.ARGUMENT_INDEX = 1
  AND cs.CONTROL_STRUCTURE_TYPE IN ["FOR", "WHILE"]

  AND (
    // Case 1: x += y, x -= y, x *= y, x /= y
    (
      c_updt.NAME IN [
        "<operator>.assignmentPlus",
        "<operator>.assignmentMinus",
        "<operator>.assignmentMultiplication",
        "<operator>.assignmentDivision"
      ]
      AND update.ARGUMENT_INDEX = 1

      // The other argument must not be a literal
      AND EXISTS {
        MATCH (c_updt)-[:ARGUMENT]->(rhs)
        WHERE rhs.ARGUMENT_INDEX = 2
          AND NOT rhs:LITERAL
      }
    )

    OR

    // Case 2: x = x + y / x = x - y / ...
    (
      c_updt.NAME = "<operator>.assignment"

      // The variable must be on the left-hand side
      AND EXISTS {
        MATCH (decl)<-[:REF]-(lhs:IDENTIFIER)<-[:ARGUMENT]-(c_updt)
        WHERE lhs.ARGUMENT_INDEX = 1
      }

      // The same variable must also appear on the right-hand side
      AND EXISTS {
        MATCH (c_updt)-[:ARGUMENT]->(rhs_root)
        WHERE rhs_root.ARGUMENT_INDEX = 2
        MATCH (rhs_root)-[:AST*0..]->(rhs_id:IDENTIFIER)-[:REF]->(decl)
      }

      // The right-hand side must contain a non-literal other argument
      AND EXISTS {
        MATCH (c_updt)-[:ARGUMENT]->(rhs_root)
        WHERE rhs_root.ARGUMENT_INDEX = 2
        MATCH (rhs_root)-[:AST*0..]->(other)
        WHERE NOT other:LITERAL
          AND NOT EXISTS {
            MATCH (other)-[:REF]->(decl)
          }
      }
    )
  )

WITH decl, cs,
     collect(DISTINCT c_updt.NAME) AS update_names,
     collect(DISTINCT c_updt) AS update_calls

WHERE all(name IN update_names WHERE name IN [
    "<operator>.assignmentPlus",
    "<operator>.assignmentMinus",
    "<operator>.assignmentMultiplication",
    "<operator>.assignmentDivision",
    "<operator>.assignment"
])

MATCH (decl)<-[:AST*]-(m:METHOD)

RETURN DISTINCT decl, m.FILENAME AS filename, update_names, update_calls
"""
