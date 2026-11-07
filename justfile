# Maintenance shortcuts. Run `just` to list them.

# The site is served from /ai-datasets/ on GitHub Pages; locally, from the root.
export BASE_PATH := env_var_or_default("BASE_PATH", "/")

# List the recipes
default:
    @just --list --unsorted

# Install dependencies and check out the published sample snapshot in .samples/
setup:
    npm install
    uv sync
    if [ ! -e .samples ]; then git fetch origin samples && git worktree add .samples samples; fi

# Sample datasets that are new or changed (all, or only the given ids)
fetch *ids:
    uv run scripts/sample.py {{ids}}

# Re-sample the given datasets even if unchanged
refetch +ids:
    uv run scripts/sample.py --force {{ids}}

# Check that every Hugging Face source still serves rows
check:
    uv run scripts/sample.py --check

# Dev server at http://localhost:4321
dev:
    npm run dev

# Build into dist/ at the site root (CI builds for /ai-datasets/)
build:
    npm run build

# Build, then serve the result locally
preview: build
    npm run preview

# Start datasets/<id>.yaml from an existing entry (e.g. `just new kitti coco`)
new id like:
    test ! -e datasets/{{id}}.yaml || { echo "datasets/{{id}}.yaml already exists"; exit 1; }
    cp datasets/{{like}}.yaml datasets/{{id}}.yaml
    @echo "Edit datasets/{{id}}.yaml, then: just fetch {{id}} && just dev"

# Run the deploy workflow on GitHub, re-fetching the given ids (or "all") even if unchanged
deploy-refetch +ids:
    gh workflow run deploy.yml -f refetch="{{ids}}"

# Remove build output and caches (keeps the .samples/ snapshot)
clean:
    rm -rf dist .astro node_modules/.astro
