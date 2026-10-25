import ast
import io
import sys
import tokenize


def check(path):
    errors = []
    with open(path, encoding="utf-8") as f:
        src = f.read()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            errors.append(f"{path}:{tok.start[0]}: comment found")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            if node.body and ast.get_docstring(node) is not None:
                errors.append(f"{path}:{node.body[0].lineno}: docstring found")
    return errors


def main():
    errors = []
    for path in sys.argv[1:]:
        errors += check(path)
    for e in errors:
        print(e)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
