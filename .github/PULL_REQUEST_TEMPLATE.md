## Summary

What changed and why? Link relevant issues.

## Verification

- [ ] A test for each fix where one would catch the bug coming back; one bug fix per commit
- [ ] The unit tests pass, Release and Debug (or failures/skips explained below)
- [ ] Persistence and cache behavior considered
- [ ] Documentation updated where behavior changes
- [ ] No private media, secrets, user settings or result caches included

Tests run, hardware used, and checks not performed:

## UI changes

Include shareable before/after screenshots when relevant.

- [ ] Checked in VML Classic and VML Modern, light and dark (`vml_appearance_tests` with `VML_UI_SNAPSHOTS` saves every tab in each)
- [ ] New controls have their role where one fits (docs/ARCHITECTURE.md, "The window's interfaces")
- [ ] A new or changed export follows docs/ARCHITECTURE.md, "Exports" (a quick grab of what is shown, or the archive)
