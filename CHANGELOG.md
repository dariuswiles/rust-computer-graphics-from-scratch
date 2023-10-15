# Changelog


## v0.1.1 - 2023-10-15
### Fixed
- Update Cargo.toml dependencies to pick up the latest versions of crates that have known security issues.
- Fix bad test of magnitude() functionality in examples/vector_math.rs that supplied points as inputs when the function only accepts vectors.
- Remove redundant calls to the format!() macro within panic!() calls.
- Remove redundant viewport_width and viewport_height fields from most raytracing examples.

## v0.1.0 - 2021-07-03
Initial release.