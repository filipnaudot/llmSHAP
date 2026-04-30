# AGENTS


# Instructions
- Never mix whitespace changes with functionality changes.
- Use the code convention specified below.


Repository instructions for code edits.

## Python Style

- When a list literal spans multiple lines, put the first item on the same line as the opening `[` whenever practical.
- Do not leave a closing `]` on its own line for these multi-line lists.
- Do not leave a closing `)` on its own line for multi-line calls, comprehensions, or expressions when the statement can be formatted cleanly without it.
- Dictionary literals are exempt from this rule. It is fine for multi-line dictionaries to use a trailing `}` on its own line.

Preferred:

```python
record_entities = [row["customer_name"],
                   row["account_owner"],
                   row["csm"]]
```

Also preferred:

```python
result = some_call(first_arg,
                   second_arg,
                   third_arg)
```

Allowed for dictionaries:

```python
FILES = {
    "company": "company.jsonl",
    "customers": "customers.jsonl",
}
```
