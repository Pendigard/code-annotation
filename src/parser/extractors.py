import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


def find_identifier_span(line: str, identifier: str) -> tuple[int, int] | None:
    pattern = rf"\b{re.escape(identifier)}\b"

    match = re.search(pattern, line)
    return None if match is None else match.span()


def find_code_span(line: str, code: str) -> tuple[int, int] | None:
    if not code:
        return None

    first_line = code.strip().splitlines()[0].strip()
    if first_line:
        index = line.find(first_line)
        if index >= 0:
            return (index, index + len(first_line))

    keyword = code.strip().split(maxsplit=1)[0] if code.strip() else ""
    if keyword:
        match = re.search(rf"\b{re.escape(keyword)}\b", line)
        if match is not None:
            return match.span()

    return None


WRITE_OPERATORS = [
    "<operator>.assignment",
    "<operator>.assignmentPlus",
    "<operator>.assignmentMinus",
    "<operator>.assignmentMultiplication",
    "<operator>.assignmentDivision",
    "<operator>.postIncrement",
    "<operator>.postDecrement",
    "<operator>.preIncrement",
    "<operator>.preDecrement",
]

COLLECTION_MUTATORS = [
    "push_back",
    "push_front",
    "pop_back",
    "pop_front",
    "insert",
    "emplace",
    "emplace_back",
    "emplace_front",
    "erase",
    "remove",
    "clear",
    "append",
]

ORGANIZER_CALLS = [
    "sort",
    "stable_sort",
    "partial_sort",
    "nth_element",
    "reverse",
    "rotate",
    "shuffle",
    "random_shuffle",
    "std::sort",
    "std::stable_sort",
    "std::partial_sort",
    "std::nth_element",
    "std::reverse",
    "std::rotate",
    "std::shuffle",
    "std::random_shuffle",
]

# -------------------
# | BASE EXTRACTORS |
# -------------------


class Neo4jConceptExtractor(ABC):
    name: str
    query: str

    def __init__(self) -> None:
        self.name = ""
        self.query = ""

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

    @abstractmethod
    def _record_to_annotation(
        self,
        record: Any,
        source_dir: Path,
    ) -> Dict[str, Any] | None:
        raise NotImplementedError


class VariableExtractor(Neo4jConceptExtractor):
    """Extract variable and parameter declarations."""

    def __init__(self, name: str, decl_only: bool) -> None:
        super().__init__()
        self.name = name
        self.decl_only = decl_only

    def _record_to_annotation(
        self,
        record: Any,
        source_dir: Path,
    ) -> Dict[str, Any] | None:
        var = record.get("var")
        filename = record.get("filename")

        if var is None or filename in (None, "<empty>"):
            return None

        variable_name = var.get("NAME")
        line_number = var.get("LINE_NUMBER")

        if variable_name is None or line_number is None:
            return None

        file_path = source_dir / filename
        if not file_path.exists():
            return None

        source = file_path.read_text(errors="replace")
        lines = source.splitlines(keepends=True)

        if line_number < 1 or line_number > len(lines):
            return None

        line = lines[line_number - 1]
        code = line.rstrip("\r\n")
        line_span = find_identifier_span(code, variable_name)

        if line_span is None:
            return None

        offset = sum(len(lines[i]) for i in range(line_number - 1))
        span = (offset + line_span[0], offset + line_span[1])
        return {
            "concept": self.name,
            "path": str(file_path),
            "variable_name": variable_name,
            "line": line_number,
            "column": var.get("COLUMN_NUMBER"),
            "code": code.strip(),
            "span_start": span[0],
            "span_end": span[1],
        }


class NodeExtractor(Neo4jConceptExtractor):
    """Extract non-variable source nodes such as control structures."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def _record_to_annotation(
        self,
        record: Any,
        source_dir: Path,
    ) -> Dict[str, Any] | None:
        node = record.get("node")
        filename = record.get("filename")

        if node is None or filename in (None, "<empty>"):
            return None

        line_number = node.get("LINE_NUMBER")
        code_value = node.get("CODE") or ""

        if line_number is None:
            return None

        file_path = source_dir / filename
        if not file_path.exists():
            return None

        source = file_path.read_text(errors="replace")
        lines = source.splitlines(keepends=True)

        if line_number < 1 or line_number > len(lines):
            return None

        line = lines[line_number - 1]
        code = line.rstrip("\r\n")
        line_span = find_code_span(code, code_value)

        if line_span is None:
            return None

        offset = sum(len(lines[i]) for i in range(line_number - 1))
        span = (offset + line_span[0], offset + line_span[1])
        return {
            "concept": self.name,
            "path": str(file_path),
            "line": line_number,
            "column": node.get("COLUMN_NUMBER"),
            "code": code.strip(),
            "span_start": span[0],
            "span_end": span[1],
        }

# ----------------------
# | CONCEPT EXTRACTORS |
# ----------------------


class FixedValue(VariableExtractor):
    """Extract variables and parameters that are never reassigned after declaration."""

    def __init__(self, name: str = "fixed_value", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)
        base_query_1: str = """
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
WHERE m.FILENAME <> '<empty>'
"""
        base_query_2: str = """
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
"""
        if self.decl_only:
            suffix = """RETURN DISTINCT decl AS var, m.FILENAME AS filename"""
        else:
            suffix = """WITH DISTINCT decl, m.FILENAME AS filename
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename
"""
        self.query = base_query_1 + suffix + " UNION " + base_query_2 + suffix


class FutureMutatedVariable(VariableExtractor):
    """Extract variables and parameters, then label them as stable or future-mutated."""

    def __init__(self, name: str = "future_mutated_variable", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query: str = """
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
"""

        if self.decl_only:
            suffix = """RETURN DISTINCT decl AS var, m.FILENAME AS filename, num_writes"""
        else:
            suffix = """WITH DISTINCT decl, m.FILENAME AS filename, num_writes
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename, num_writes
"""

        self.query = base_query + suffix

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


class Gatherer(VariableExtractor):
    """Extract variables that are used to accumulate state across method calls."""

    def __init__(self, name: str = "gatherer", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query: str = """
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
"""

        if self.decl_only:
            suffix = """RETURN DISTINCT decl AS var, m.FILENAME AS filename, update_names, update_calls"""
        else:
            suffix = """WITH DISTINCT decl, m.FILENAME AS filename, update_names, update_calls
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename, update_names, update_calls
"""
        self.query = base_query + suffix


class Stepper(VariableExtractor):
    """Extract variables that are used to step through a sequence of values."""

    def __init__(self, name: str = "stepper", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        if self.decl_only:
            suffix = """
RETURN DISTINCT decl AS var, m.FILENAME AS filename, update_names
"""
        else:
            suffix = """
WITH DISTINCT decl, m.FILENAME AS filename, update_names

MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename, update_names
"""

        base_query: str = """MATCH (cs:CONTROL_STRUCTURE)
WHERE cs.CONTROL_STRUCTURE_TYPE IN ["FOR", "WHILE"]

// 1. Variable is used in the loop condition
MATCH (cs)-[:CONDITION]->(cond_root)
MATCH (cond_root)-[:AST*0..]->(cond_id:IDENTIFIER)-[:REF]->(decl:LOCAL)

// 2. Find all updates to this variable inside the loop body
MATCH (cs)-[:AST*1..]->(c_updt:CALL)
MATCH (decl)<-[:REF]-(update_id:IDENTIFIER)<-[:ARGUMENT]-(c_updt)
WHERE update_id.ARGUMENT_INDEX = 1 // Left hand side of the writing operation

// Exclude the initial declaration assignment if it happens to sit exactly on the line/col of the update
AND (decl.LINE_NUMBER <> update_id.LINE_NUMBER OR decl.COLUMN_NUMBER <> update_id.COLUMN_NUMBER)

// 3. Ensure updates are strictly using literals (e.g., x += 1 or x = x + 2)
// Case A: Unary or compound assignment like x++ or x += 1
// Case B: Explicit assignment using a literal/constant like x = x + 1
AND (
  // Case A: The update is a post/pre increment/decrement OR compound assignment with a literal
  (c_updt.NAME IN [
    "<operator>.postIncrement", "<operator>.preIncrement",
    "<operator>.postDecrement", "<operator>.preDecrement"
  ])
  OR
  (c_updt.NAME IN [
     "<operator>.assignmentPlus", "<operator>.assignmentMinus",
     "<operator>.assignmentMultiplication", "<operator>.assignmentDivision"
   ]
   AND EXISTS {
     MATCH (c_updt)-[:ARGUMENT]->(rhs:LITERAL)
     WHERE rhs.ARGUMENT_INDEX = 2
   })
  OR
  // Case B: Explicit assignment (x = x + 1)
  (c_updt.NAME = "<operator>.assignment"
   AND EXISTS {
     MATCH (c_updt)-[:ARGUMENT]->(rhs_root:CALL)
     WHERE rhs_root.ARGUMENT_INDEX = 2
       AND rhs_root.NAME IN ["<operator>.addition", "<operator>.subtraction", "<operator>.multiplication", "<operator>.division"]
     // One side of the math operation is the variable itself, the other is a literal
     AND EXISTS { MATCH (rhs_root)-[:ARGUMENT]->(mid:IDENTIFIER)-[:REF]->(decl) }
     AND EXISTS { MATCH (rhs_root)-[:ARGUMENT]->(lit:LITERAL) }
   })
)

// 4. Aggregation to guarantee homogeneity (All updates must be the same operator)
WITH decl, cs, 
     collect(DISTINCT c_updt.NAME) AS update_names,
     collect(DISTINCT c_updt) AS update_calls

// Ensure there is exactly 1 type of update operator used across the board
WHERE size(update_names) = 1

// Tie it back to the containing method to output the correct file path
MATCH (decl)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
"""
        self.query = base_query + suffix


class Walker(VariableExtractor):
    """Extract variables that traverse a collection by index, iterator, or next-like calls."""

    def __init__(self, name: str = "walker", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query = """
MATCH (cs:CONTROL_STRUCTURE)
WHERE cs.CONTROL_STRUCTURE_TYPE IN ["FOR", "WHILE"]
MATCH (cs)-[:CONDITION]->(cond)
MATCH (cond)-[:AST*0..]->(:IDENTIFIER)-[:REF]->(decl:LOCAL)
WHERE EXISTS {
  MATCH (cs)-[:AST*1..]->(idx:CALL)
  WHERE idx.NAME IN ["<operator>.indirectIndexAccess", "<operator>.indirection"]
  MATCH (idx)-[:AST*0..]->(:IDENTIFIER)-[:REF]->(decl)
}
OR EXISTS {
  MATCH (cs)-[:AST*1..]->(use:IDENTIFIER)-[:REF]->(decl)
  MATCH (use)<-[:ARGUMENT]-(call:CALL)
  WHERE call.NAME IN ["next", "hasNext", "begin", "end", "find"]
}
MATCH (decl)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
"""
        if self.decl_only:
            suffix = "RETURN DISTINCT decl AS var, m.FILENAME AS filename"
        else:
            suffix = """
WITH DISTINCT decl, m.FILENAME AS filename
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename
"""
        self.query = base_query + suffix


class MostRecentHolder(VariableExtractor):
    """Extract locals overwritten inside a loop with the latest non-self value."""

    def __init__(self, name: str = "most_recent_holder", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query = """
MATCH (decl:LOCAL)<-[:REF]-(lhs:IDENTIFIER)<-[:ARGUMENT]-(assign:CALL {NAME:"<operator>.assignment"})
MATCH (assign)<-[:AST*]-(cs:CONTROL_STRUCTURE)
WHERE cs.CONTROL_STRUCTURE_TYPE IN ["FOR", "WHILE", "DO"]
  AND lhs.ARGUMENT_INDEX = 1
  AND (decl.LINE_NUMBER <> lhs.LINE_NUMBER OR decl.COLUMN_NUMBER <> lhs.COLUMN_NUMBER)
  AND EXISTS {
    MATCH (assign)-[:ARGUMENT]->(rhs)
    WHERE rhs.ARGUMENT_INDEX = 2
      AND NOT EXISTS { MATCH (rhs)-[:AST*0..]->(:IDENTIFIER)-[:REF]->(decl) }
      AND (
        rhs:IDENTIFIER OR
        EXISTS { MATCH (rhs)-[:AST*0..]->(:IDENTIFIER) } OR
        EXISTS { MATCH (rhs)-[:AST*0..]->(c:CALL) WHERE c.NAME IN ["next", "read", "getline", "get", "front", "back", "at"] }
      )
  }
MATCH (decl)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
"""
        if self.decl_only:
            suffix = "RETURN DISTINCT decl AS var, m.FILENAME AS filename"
        else:
            suffix = """
WITH DISTINCT decl, m.FILENAME AS filename
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename
"""
        self.query = base_query + suffix


class MostWantedHolder(VariableExtractor):
    """Extract best-so-far variables updated conditionally inside loops."""

    def __init__(self, name: str = "most_wanted_holder", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query = """
MATCH (decl:LOCAL)<-[:REF]-(lhs:IDENTIFIER)<-[:ARGUMENT]-(assign:CALL {NAME:"<operator>.assignment"})
MATCH (assign)<-[:AST*]-(ifcs:CONTROL_STRUCTURE {CONTROL_STRUCTURE_TYPE:"IF"})
MATCH (ifcs)<-[:AST*]-(loop:CONTROL_STRUCTURE)
WHERE loop.CONTROL_STRUCTURE_TYPE IN ["FOR", "WHILE", "DO"]
  AND lhs.ARGUMENT_INDEX = 1
  AND (decl.LINE_NUMBER <> lhs.LINE_NUMBER OR decl.COLUMN_NUMBER <> lhs.COLUMN_NUMBER)
  AND (
    EXISTS {
      MATCH (ifcs)-[:CONDITION]->(cond)
      MATCH (cond)-[:AST*0..]->(cmp:CALL)
      WHERE cmp.NAME IN [
        "<operator>.lessThan", "<operator>.lessEqualsThan",
        "<operator>.greaterThan", "<operator>.greaterEqualsThan"
      ]
      MATCH (cmp)-[:AST*0..]->(:IDENTIFIER)-[:REF]->(decl)
    }
    OR assign.CODE =~ '(?i).*\\b(min|max)\\b.*'
  )
MATCH (decl)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
"""
        if self.decl_only:
            suffix = "RETURN DISTINCT decl AS var, m.FILENAME AS filename"
        else:
            suffix = """
WITH DISTINCT decl, m.FILENAME AS filename
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename
"""
        self.query = base_query + suffix


class OneWayFlag(VariableExtractor):
    """Extract boolean-like locals flipped once and never restored."""

    def __init__(self, name: str = "one_way_flag", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query = """
MATCH (decl:LOCAL)<-[:REF]-(init_lhs:IDENTIFIER)<-[:ARGUMENT]-(init:CALL {NAME:"<operator>.assignment"})
MATCH (init)-[:ARGUMENT]->(init_lit:LITERAL)
WHERE init_lhs.ARGUMENT_INDEX = 1
  AND init_lit.ARGUMENT_INDEX = 2
  AND init_lit.CODE IN ["true", "false", "0", "1"]
MATCH (decl)<-[:REF]-(write_lhs:IDENTIFIER)<-[:ARGUMENT]-(write:CALL {NAME:"<operator>.assignment"})
MATCH (write)-[:ARGUMENT]->(write_lit:LITERAL)
WHERE write_lhs.ARGUMENT_INDEX = 1
  AND write_lit.ARGUMENT_INDEX = 2
  AND (decl.LINE_NUMBER <> write_lhs.LINE_NUMBER OR decl.COLUMN_NUMBER <> write_lhs.COLUMN_NUMBER)
  AND (
    (init_lit.CODE IN ["false", "0"] AND write_lit.CODE IN ["true", "1"]) OR
    (init_lit.CODE IN ["true", "1"] AND write_lit.CODE IN ["false", "0"])
  )
  AND NOT EXISTS {
    MATCH (decl)<-[:REF]-(restore_lhs:IDENTIFIER)<-[:ARGUMENT]-(restore:CALL {NAME:"<operator>.assignment"})
    MATCH (restore)-[:ARGUMENT]->(restore_lit:LITERAL)
    WHERE restore_lhs.ARGUMENT_INDEX = 1
      AND restore_lit.ARGUMENT_INDEX = 2
      AND restore_lit.CODE = init_lit.CODE
      AND restore.LINE_NUMBER > write.LINE_NUMBER
  }
MATCH (decl)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
"""
        if self.decl_only:
            suffix = "RETURN DISTINCT decl AS var, m.FILENAME AS filename"
        else:
            suffix = """
WITH DISTINCT decl, m.FILENAME AS filename
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename
"""
        self.query = base_query + suffix


class Organizer(VariableExtractor):
    """Extract collections rearranged in place without size-changing calls."""

    def __init__(self, name: str = "organizer", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query = f"""
MATCH (decl:LOCAL|METHOD_PARAMETER_IN)<-[:REF]-(use:IDENTIFIER)
MATCH (use)<-[:AST*0..]-(arg)
MATCH (arg)<-[:ARGUMENT]-(call:CALL)
WHERE call.NAME IN {ORGANIZER_CALLS}
  AND NOT EXISTS {{
    MATCH (decl)<-[:REF]-(mut:IDENTIFIER)
    MATCH (mut)<-[:AST*0..]-(recv)
    MATCH (recv)<-[:ARGUMENT]-(mut_call:CALL)
    WHERE mut_call.NAME IN {COLLECTION_MUTATORS}
  }}
MATCH (decl)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
"""
        if self.decl_only:
            suffix = "RETURN DISTINCT decl AS var, m.FILENAME AS filename"
        else:
            suffix = """
WITH DISTINCT decl, m.FILENAME AS filename
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename
"""
        self.query = base_query + suffix


class Container(VariableExtractor):
    """Extract collections that have elements added or removed."""

    def __init__(self, name: str = "container", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query = f"""
MATCH (decl:LOCAL|METHOD_PARAMETER_IN)<-[:REF]-(recv_id:IDENTIFIER)
MATCH (recv_id)<-[:AST*0..]-(recv)
MATCH (recv)<-[:ARGUMENT]-(call:CALL)
WHERE call.NAME IN {COLLECTION_MUTATORS}
  AND coalesce(recv.ARGUMENT_INDEX, recv_id.ARGUMENT_INDEX) = 0
MATCH (decl)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
"""
        if self.decl_only:
            suffix = "RETURN DISTINCT decl AS var, m.FILENAME AS filename"
        else:
            suffix = """
WITH DISTINCT decl, m.FILENAME AS filename
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename
"""
        self.query = base_query + suffix


class VariableDependent(VariableExtractor):
    """Extract variables assigned from an expression containing another variable."""

    def __init__(self, name: str = "variable_dependent", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query = """
MATCH (decl:LOCAL|METHOD_PARAMETER_IN)<-[:REF]-(lhs:IDENTIFIER)<-[:ARGUMENT]-(assign:CALL {NAME:"<operator>.assignment"})
MATCH (assign)-[:ARGUMENT]->(rhs)
WHERE lhs.ARGUMENT_INDEX = 1
  AND rhs.ARGUMENT_INDEX = 2
  AND EXISTS {
    MATCH (rhs)-[:AST*0..]->(:IDENTIFIER)-[:REF]->(other:LOCAL|METHOD_PARAMETER_IN)
    WHERE other <> decl
  }
MATCH (decl)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
"""
        if self.decl_only:
            suffix = "RETURN DISTINCT decl AS var, m.FILENAME AS filename"
        else:
            suffix = """
WITH DISTINCT decl, m.FILENAME AS filename
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename
"""
        self.query = base_query + suffix


class Follower(VariableExtractor):
    """Extract variables that store another variable before that variable changes."""

    def __init__(self, name: str = "follower", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query = f"""
MATCH (decl:LOCAL)<-[:REF]-(lhs:IDENTIFIER)<-[:ARGUMENT]-(assign:CALL {{NAME:"<operator>.assignment"}})
MATCH (assign)-[:ARGUMENT]->(rhs_id:IDENTIFIER)-[:REF]->(leader:LOCAL|METHOD_PARAMETER_IN)
WHERE lhs.ARGUMENT_INDEX = 1
  AND rhs_id.ARGUMENT_INDEX = 2
  AND leader <> decl
  AND EXISTS {{
    MATCH (leader)<-[:REF]-(leader_lhs:IDENTIFIER)<-[:ARGUMENT]-(later_write:CALL)
    WHERE later_write.NAME IN {WRITE_OPERATORS}
      AND leader_lhs.ARGUMENT_INDEX = 1
      AND later_write.LINE_NUMBER > assign.LINE_NUMBER
  }}
MATCH (decl)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
"""
        if self.decl_only:
            suffix = "RETURN DISTINCT decl AS var, m.FILENAME AS filename"
        else:
            suffix = """
WITH DISTINCT decl, m.FILENAME AS filename
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename
"""
        self.query = base_query + suffix


class FutureDependency(VariableExtractor):
    """Extract values later read by another computation, condition, call, or return."""

    def __init__(self, name: str = "future_dependency", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query = f"""
MATCH (decl:LOCAL|METHOD_PARAMETER_IN)<-[:REF]-(use:IDENTIFIER)
WHERE use.LINE_NUMBER > decl.LINE_NUMBER
  AND NOT EXISTS {{
    MATCH (use)<-[:ARGUMENT]-(write:CALL)
    WHERE write.NAME IN {WRITE_OPERATORS}
      AND use.ARGUMENT_INDEX = 1
  }}
MATCH (decl)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
"""
        if self.decl_only:
            suffix = "RETURN DISTINCT decl AS var, m.FILENAME AS filename"
        else:
            suffix = """
WITH DISTINCT decl, m.FILENAME AS filename
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename
"""
        self.query = base_query + suffix


class FutureReturnDependency(VariableExtractor):
    """Extract variables whose value is used in a later return expression."""

    def __init__(self, name: str = "future_return_dependency", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query = """
MATCH (ret:RETURN)-[:AST*0..]->(:IDENTIFIER)-[:REF]->(decl:LOCAL|METHOD_PARAMETER_IN)
WHERE ret.LINE_NUMBER >= decl.LINE_NUMBER
MATCH (decl)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
"""
        if self.decl_only:
            suffix = "RETURN DISTINCT decl AS var, m.FILENAME AS filename"
        else:
            suffix = """
WITH DISTINCT decl, m.FILENAME AS filename
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename
"""
        self.query = base_query + suffix


class VariableUsedInFunction(VariableExtractor):
    """Extract variables passed as arguments to non-operator calls."""

    def __init__(self, name: str = "variable_used_in_function", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query = """
MATCH (decl:LOCAL|METHOD_PARAMETER_IN)<-[:REF]-(arg_id:IDENTIFIER)
MATCH (arg_id)<-[:AST*0..]-(arg)
MATCH (arg)<-[:ARGUMENT]-(call:CALL)
WHERE call.NAME IS NOT NULL
  AND NOT call.NAME STARTS WITH "<operator>."
  AND coalesce(arg.ARGUMENT_INDEX, arg_id.ARGUMENT_INDEX) >= 1
MATCH (decl)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
"""
        if self.decl_only:
            suffix = "RETURN DISTINCT decl AS var, m.FILENAME AS filename"
        else:
            suffix = """
WITH DISTINCT decl, m.FILENAME AS filename
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename
"""
        self.query = base_query + suffix


class Temporary(VariableExtractor):
    """Extract temporary swap/short-lived variables."""

    def __init__(self, name: str = "temporary", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query = """
MATCH (decl:LOCAL)<-[:REF]-(tmp_lhs:IDENTIFIER)<-[:ARGUMENT]-(tmp_assign:CALL {NAME:"<operator>.assignment"})
MATCH (tmp_assign)-[:ARGUMENT]->(first_rhs:IDENTIFIER)-[:REF]->(a:LOCAL|METHOD_PARAMETER_IN)
MATCH (a)<-[:REF]-(a_lhs:IDENTIFIER)<-[:ARGUMENT]-(a_assign:CALL {NAME:"<operator>.assignment"})
MATCH (a_assign)-[:ARGUMENT]->(:IDENTIFIER)-[:REF]->(b:LOCAL|METHOD_PARAMETER_IN)
MATCH (b)<-[:REF]-(b_lhs:IDENTIFIER)<-[:ARGUMENT]-(b_assign:CALL {NAME:"<operator>.assignment"})
MATCH (b_assign)-[:ARGUMENT]->(:IDENTIFIER)-[:REF]->(decl)
WHERE tmp_lhs.ARGUMENT_INDEX = 1
  AND first_rhs.ARGUMENT_INDEX = 2
  AND a_lhs.ARGUMENT_INDEX = 1
  AND b_lhs.ARGUMENT_INDEX = 1
  AND tmp_assign.LINE_NUMBER <= a_assign.LINE_NUMBER
  AND a_assign.LINE_NUMBER <= b_assign.LINE_NUMBER
  AND b <> a
MATCH (decl)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
"""
        if self.decl_only:
            suffix = "RETURN DISTINCT decl AS var, m.FILENAME AS filename"
        else:
            suffix = """
WITH DISTINCT decl, m.FILENAME AS filename
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename
"""
        self.query = base_query + suffix


class FutureBranchDependency(VariableExtractor):
    """Extract variables later used to control an if/switch condition."""

    def __init__(self, name: str = "future_branch_dependency", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query = """
MATCH (cs:CONTROL_STRUCTURE)
WHERE cs.CONTROL_STRUCTURE_TYPE IN ["IF", "SWITCH"]
MATCH (cs)-[:CONDITION]->(cond)
MATCH (cond)-[:AST*0..]->(:IDENTIFIER)-[:REF]->(decl:LOCAL|METHOD_PARAMETER_IN)
WHERE cs.LINE_NUMBER >= decl.LINE_NUMBER
MATCH (decl)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
"""
        if self.decl_only:
            suffix = "RETURN DISTINCT decl AS var, m.FILENAME AS filename"
        else:
            suffix = """
WITH DISTINCT decl, m.FILENAME AS filename
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename
"""
        self.query = base_query + suffix


class SingleControlFlow(NodeExtractor):
    """Extract if statements with no explicit alternative branch."""

    def __init__(self, name: str = "single_control_flow") -> None:
        super().__init__(name=name)
        self.query = """
MATCH (node:CONTROL_STRUCTURE {CONTROL_STRUCTURE_TYPE:"IF"})
WHERE NOT node.CODE =~ '(?s).*\\belse\\b.*'
MATCH (node)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
RETURN DISTINCT node, m.FILENAME AS filename
"""


class BinaryControlFlow(NodeExtractor):
    """Extract if/else statements with exactly one alternative branch."""

    def __init__(self, name: str = "binary_control_flow") -> None:
        super().__init__(name=name)
        self.query = """
MATCH (node:CONTROL_STRUCTURE {CONTROL_STRUCTURE_TYPE:"IF"})
WHERE node.CODE =~ '(?s).*\\belse\\b.*'
  AND NOT node.CODE =~ '(?s).*\\belse\\s+if\\b.*'
MATCH (node)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
RETURN DISTINCT node, m.FILENAME AS filename
"""


class NCaseControlFlow(NodeExtractor):
    """Extract switch or else-if chains with multiple alternatives."""

    def __init__(self, name: str = "n_case_control_flow") -> None:
        super().__init__(name=name)
        self.query = """
MATCH (node:CONTROL_STRUCTURE)
WHERE node.CONTROL_STRUCTURE_TYPE = "SWITCH"
   OR (node.CONTROL_STRUCTURE_TYPE = "IF" AND node.CODE =~ '(?s).*\\belse\\s+if\\b.*')
MATCH (node)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
RETURN DISTINCT node, m.FILENAME AS filename
"""


class FutureLoopEntry(NodeExtractor):
    """Extract loop control structures as entry points to iterative execution."""

    def __init__(self, name: str = "future_loop_entry") -> None:
        super().__init__(name=name)
        self.query = """
MATCH (node:CONTROL_STRUCTURE)
WHERE node.CONTROL_STRUCTURE_TYPE IN ["FOR", "WHILE", "DO"]
MATCH (node)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
RETURN DISTINCT node, m.FILENAME AS filename
"""


class FutureRecursiveCall(VariableExtractor):
    """Extract call arguments that flow into a recursive invocation."""

    def __init__(self, name: str = "future_recursive_call", decl_only: bool = False) -> None:
        super().__init__(name=name, decl_only=decl_only)

        base_query = """
MATCH (call:CALL)<-[:AST*]-(m:METHOD)
WHERE call.METHOD_FULL_NAME = m.FULL_NAME OR call.NAME = m.NAME
MATCH (call)-[:ARGUMENT]->(arg)
MATCH (arg)-[:AST*0..]->(:IDENTIFIER)-[:REF]->(decl:LOCAL|METHOD_PARAMETER_IN)
WHERE m.FILENAME <> '<empty>'
"""
        if self.decl_only:
            suffix = "RETURN DISTINCT decl AS var, m.FILENAME AS filename"
        else:
            suffix = """
WITH DISTINCT decl, m.FILENAME AS filename
MATCH (var:IDENTIFIER)-[:REF]->(decl)
RETURN var, filename
"""
        self.query = base_query + suffix


class FutureBranchMerge(NodeExtractor):
    """Extract branch structures whose distinct paths reconverge later in CFG."""

    def __init__(self, name: str = "future_branch_merge") -> None:
        super().__init__(name=name)
        self.query = """
MATCH (node:CONTROL_STRUCTURE)
WHERE node.CONTROL_STRUCTURE_TYPE IN ["IF", "SWITCH"]
  AND EXISTS {
    MATCH (node)-[:CDG]->(a)
    MATCH (node)-[:CDG]->(b)
    WHERE elementId(a) < elementId(b)
    MATCH (a)-[:POST_DOMINATE]->(merge)
    MATCH (b)-[:POST_DOMINATE]->(merge)
    WHERE merge.LINE_NUMBER > node.LINE_NUMBER
  }
MATCH (node)<-[:AST*]-(m:METHOD)
WHERE m.FILENAME <> '<empty>'
RETURN DISTINCT node, m.FILENAME AS filename
"""
