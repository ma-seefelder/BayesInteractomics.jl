# Expected TR/DDI values — 2-species metalearner fixture

This directory holds a minimal real-schema 2-species fixture (9606 human + 10090
mouse), each species contributing exactly one within-species protein pair. It is
the input the per-species feature-lookup and on-the-fly featuriser tests
run against.

The 6 production-schema TR/DDI columns are, in canonical order
(`METALEARNER_FEATURE_LOOKUP_COLUMNS`):

```
neighborhood_tr, experiments_tr, database_tr, textmining_tr, ddi_n_known, ddi_has_known
```

## Fixture source files

| File | Schema |
|------|--------|
| `<sp>_mini.aliases.txt` | STRING aliases: `#string_protein_id`, `alias`, `source` (tab-separated). One `Ensembl_UniProt` row per protein → UniProt accession. |
| `<sp>_mini.links.full.txt` | STRING `links.full` v12: 16-token **space-separated** `STRING_DETAILED_COLS` header + one scored row per ordered direction of the pair. |
| `uniprot_mini_sl_pfam.tsv` | UniProt stream: `accession`, `id`, `reviewed`, `cc_subcellular_location`, `xref_pfam` (tab-separated). Shared across species. |
| `3did_mini_flat.txt` | 3did flat catalogue: `#=ID` rows carrying one `(PFxxxx.NN@Pfam <ws> PFyyyy.NN@Pfam)` Pfam-pair each. Shared, species-agnostic. |

`STRING_DETAILED_COLS` (16, verbatim — must equal the first line of every
`<sp>_mini.links.full.txt`):

```
protein1 protein2 neighborhood neighborhood_transferred fusion cooccurence homology coexpression coexpression_transferred experiments experiments_transferred database database_transferred textmining textmining_transferred combined_score
```

The 4 transferred channels extracted (`STRING_TRANSFERRED_COLS`) are columns
4, 11, 13, 15 → `neighborhood_transferred, experiments_transferred,
database_transferred, textmining_transferred`.

## Hand-computed expectations

Values verified against the species-neutral spike-009 builders
(`load_string_to_uniprot` → `load_string_detailed` /
`build_string_transferred_features` → `load_3did_catalogue` +
`build_protein_pfam_map` + `build_ddi_features`) fed the fixture files directly.

### 9606 (human) pair

- STRING IDs: `9606.ENSP00000A1` ↔ `9606.ENSP00000A2`
- UniProt: `P11111` (Pfam `PF00001`) ↔ `P22222` (Pfam `PF00002`)
- 3did contains `(PF00001, PF00002)` → 1 known DDI

| column | value |
|--------|-------|
| neighborhood_tr | 150.0 |
| experiments_tr  | 200.0 |
| database_tr     | 300.0 |
| textmining_tr   | 400.0 |
| ddi_n_known     | 1.0   |
| ddi_has_known   | 1.0   |

### 10090 (mouse) pair — sparse-species featurised (non-zero)

- STRING IDs: `10090.ENSMUSP00000B1` ↔ `10090.ENSMUSP00000B2`
- UniProt: `Q33333` (Pfam `PF00003`) ↔ `Q44444` (Pfam `PF00004`)
- 3did contains `(PF00003, PF00004)` → 1 known DDI

| column | value |
|--------|-------|
| neighborhood_tr | 110.0 |
| experiments_tr  | 120.0 |
| database_tr     | 130.0 |
| textmining_tr   | 140.0 |
| ddi_n_known     | 1.0   |
| ddi_has_known   | 1.0   |

## Canonical key

Both pairs use the lexicographic `(min, max)` STRING-ID key
(`_canonical_pair_key`); the taxon prefix is already part of the STRING ID, so
no separate species field is needed in the key (Pitfall 1).

- 9606 key: `("9606.ENSP00000A1", "9606.ENSP00000A2")`
- 10090 key: `("10090.ENSMUSP00000B1", "10090.ENSMUSP00000B2")`
