#!/usr/bin/env bash
set -euo pipefail

# Extract version from pyproject.toml without executing arbitrary code:
# Requires python>=3.11 for tomllib, but that's OK in CI.
VERSION="$(python -c 'import tomllib;print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')"
TAG="v${VERSION}"

# Strict matching: look for an assignment like __version__ = "x.y.z"
grep -Eq "__version__\s*=\s*[\"']${VERSION}[\"']" amep/_version.py || {
  echo "amep/_version.py does not define __version__ as ${VERSION}"
  exit 1
}

# doc/source/conf.py often contains 'release = "x.y.z"' or 'version = "x.y"'
grep -Eq "release\s*=\s*[\"']${VERSION}[\"']" doc/source/conf.py || {
  echo "doc/source/conf.py does not define release as ${VERSION}"
  exit 1
}

grep -Fq "${VERSION}" CHANGELOG.md || {
  echo "CHANGELOG.md does not mention ${VERSION}"
  exit 1
}

# Emit outputs safely
{
  echo "VERSION=${VERSION}"
  echo "TAG=${TAG}"
} >> "$GITHUB_ENV"

{
  echo "version=${VERSION}"
  echo "tag=${TAG}"
} >> "$GITHUB_OUTPUT"


echo "Validated release version ${VERSION} (${TAG})"
