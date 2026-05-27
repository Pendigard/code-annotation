import json
from dataclasses import dataclass
from typing import Dict, List, Protocol, Any
import ast
from cpgqls_client import CPGQLSClient


def run_query(client: CPGQLSClient, query: str) -> str:
    result = client.execute(query)

    if isinstance(result, dict):
        stderr = result.get("stderr")
        if stderr:
            raise RuntimeError(stderr)
        return result.get("stdout", "")

    return str(result)

def parse_json_lines(raw: str) -> List[Dict[str, Any]]:
    annotations = []

    for line in raw.splitlines():
        line = line.strip().rstrip(",")

        # Keep only Scala string literals
        if line.startswith('"') and line.endswith('"'):
            try:
                # Decode Scala/Python-style escaped string
                decoded = ast.literal_eval(line)

                annotations.append(json.loads(decoded))

            except Exception as e:
                print(f"[WARN] Failed to parse line: {line}")
                print(e)

    return annotations

# ----------------------------
# Concept extractor interface
# ----------------------------

class ConceptExtractor(Protocol):
    name: str

    def query(self) -> str:
        ...

    def parse(self, raw: str) -> List[Dict[str, Any]]:
        ...

@dataclass
class JoernConceptExtractor:
    """
    Base class for simple Joern concept extractors.
    Subclasses only need to implement `query`.
    """
    name: str

    def query(self) -> str:
        raise NotImplementedError

    def parse(self, raw: str) -> List[Dict[str, Any]]:
        return parse_json_lines(raw)

    def run(self, client: CPGQLSClient) -> List[Dict[str, Any]]:
        raw = run_query(client, self.query())
        return self.parse(raw)
    
# ----------------------------
# Specific concept extractors
# ----------------------------

@dataclass
class ArithmeticExpressionExtractor(JoernConceptExtractor):
    name: str = "arithmetic_expression"

    def query(self) -> str:
        return r'''
        cpg.call
        .filter(c =>
            c.name == "<operator>.addition" ||
            c.name == "<operator>.subtraction" ||
            c.name == "<operator>.multiplication" ||
            c.name == "<operator>.division" ||
            c.name == "<operator>.modulo"
        )
        .map(c => {
            val code = c.code.replace("\\", "\\\\").replace("\"", "\\\"")
            val file = c.location.filename.replace("\\", "\\\\").replace("\"", "\\\"")
            val method = c.method.name.replace("\\", "\\\\").replace("\"", "\\\"")

            s"""{"concept":"arithmetic_expression","operator":"${c.name}","code":"${code}","file":"${file}","line":${c.lineNumber.getOrElse(-1)},"column":${c.columnNumber.getOrElse(-1)},"method":"${method}"}"""
        })
        .l
        '''
    
@dataclass
class LoopBlockExtractor(JoernConceptExtractor):
    name: str = "loop_block"

    def query(self) -> str:
        return r'''
        cpg.controlStructure
        .filter(c =>
            c.controlStructureType == "FOR" ||
            c.controlStructureType == "WHILE" ||
            c.controlStructureType == "DO"
        )
        .map(c => {
            val code = c.code.replace("\\", "\\\\").replace("\"", "\\\"")
            val file = c.location.filename.replace("\\", "\\\\").replace("\"", "")
            val method = c.method.name.replace("\\", "\\\\").replace("\"", "")

            s"""{"concept":"loop_block","type":"${c.controlStructureType}","code":"${code}","file":"${file}","start_line":${c.lineNumber.getOrElse(-1)},"start_column":${c.columnNumber.getOrElse(-1)},"end_line":${c.lineNumberEnd.getOrElse(-1)},"end_column":${c.columnNumberEnd.getOrElse(-1)},"method":"${method}"}"""
        })
        .l
        '''