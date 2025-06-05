# CTE Metrics Quick Test

A minimal setup to compute Cross Topological Entropy (CTE) and JSD-based similarity between two code snippets using Tree-Sitter.

---

## Requirements

* **Python 3.7+**
* Install dependencies:

  ```bash
  pip install tree_sitter_language_pack numpy nltk
  ```

  > *Optional:*
  >
  > * `pandas` if you plan to collect results in DataFrames
  > * To download NLTK tokenizers (if not already present):
  >
  >   ```bash
  >   python -m nltk.downloader punkt
  >   ```

---

## Quick Start

1. **Save the code** in a file named `cte_metrics.py`. This file should contain all the functions defined previously:

   * `_normalize_vector`
   * `kullback_leibler_divergence`
   * `jensen_shannon_divergence`
   * `extract_subtrees`
   * `extract_subtrees_withvalue`
   * `compute_cte_norm_struct`
   * `compute_cte_jsd_struct`
   * `compute_cte_norm_value`
   * `compute_cte_jsd_value`

2. **Create a short test script** (e.g., `test_cte.py`) with the following content:

   ```python
   from cte_metrics import (
       compute_cte_norm_struct,
       compute_cte_jsd_struct,
       compute_cte_norm_value,
       compute_cte_jsd_value,
   )

   code1 = """
   if x >= 0:
       sign = "non-negative"
   else:
       sign = "negative"
   print(sign)
   """

   code2 = """
   if x >= 0:
       sign = "non-negative"
       print(sign)
   else:
       sign = "negative"
       print(sign)
   """

   lang = "python"
   max_depth = 10

   print("CTE-JSD Struct:", compute_cte_jsd_struct(lang, code1, code2, max_depth))
   print("CTE-Norm Struct:", compute_cte_norm_struct(lang, code1, code2, max_depth))
   print("CTE-JSD Value:", compute_cte_jsd_value(lang, code1, code2, max_depth))
   print("CTE-Norm Value:", compute_cte_norm_value(lang, code1, code2, max_depth))
   ```

3. **Run the test**:

   ```bash
   python test_cte.py
   ```

You should see four floating-point scores printed to the console:

* Structural JSD similarity (CTE-JSD Struct)
* Structural normalized CTE (CTE-Norm Struct)
* Value-based JSD similarity (CTE-JSD Value)
* Value-based normalized CTE (CTE-Norm Value)

---

## Configuration Options

* **`lang`**
  Specify the language tag understood by Tree-Sitter (e.g., `"python"`, `"sql"`, `"javascript"`).

* **`max_depth`**
  Determines how deep the AST traversal goes. Increase for more detailed subtree extraction; decrease to speed up comparisons.

* **`eps` (smoothing constant)**
  Defaults to `1e-10`. Prevents zero-probability issues when computing entropy/divergence.

---

## Notes

* All distributions are automatically normalized under the hood using `_normalize_vector`.
* The JSD-based functions compute

  ```text
  JS Divergence = 0.5 * [KL(P || M) + KL(Q || M)],  where M = (P + Q)/2
  ```

  and return `1 – JSD` to yield a similarity score bounded in \[0, 1].
* If any divergence calculation yields `NaN` (due to numerical issues), the functions will print a warning and default that JSD to `1.0` (resulting in similarity = 0.0).
* To compare code in other languages, install the appropriate Tree-Sitter grammars via `tree_sitter_language_pack` and set `lang` accordingly.

---

## Example Output

```text
CTE-JSD Struct:     0.871320
CTE-Norm Struct:    0.833333
CTE-JSD Value:      0.785398
CTE-Norm Value:     0.800000
```

*(Values may vary slightly depending on AST depth and implementation details.)*
