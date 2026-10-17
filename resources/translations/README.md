# Translations

One file per language, `<code>.json`, named by the codes in `cpp/app/i18n.cpp`'s `all_languages`:

```json
{
  "strings": {"Settings saved.": "Einstellungen gespeichert."},
  "plurals": {"{count} video failed": ["{count} Video fehlgeschlagen", "{count} Videos fehlgeschlagen"]}
}
```

- The keys are the English text exactly as the code has it, `{placeholders}` included. A translation keeps every
  placeholder, with its format (`{fps:.1f}`), and may move it.
- `plurals` lists one form per plural category of the language, in the order `plural_index` in `cpp/app/i18n.cpp` gives
  (one form for Chinese, Japanese, Korean, Thai, Vietnamese and Indonesian; three for Russian, Ukrainian, Polish
  and Czech; six for Arabic; two for the rest).
- Metric names (VMAF, SSIMULACRA2, CVVDP...), FFmpeg, Vship, GStreamer, D3D11, GPU and CPU stay as they are.
- Only the window is translated. The log, saved results and exported files stay in English.

`python scripts/i18n_catalog.py check` lists what each catalog lacks after the English text changes. The window's
tests hold its text in each language to `cpp/tests/reference/i18n.json`, made by the Python app at the port: a change
to a text it lists, or to that text's translation, edits it too.
