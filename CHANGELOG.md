# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### What's Changed

- Updated Dockerfile to create a non-root user and set permissions.
- Included more logs to check the lesions segmentation inference process.
- Added new `SITKCastd` tranform to cast predicted TTB to `uint8` as expected by the challenge.
- Added initial inference code for Task007-PBD.
