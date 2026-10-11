# Fixture shop

A small storefront used by the token benchmark. The Go service loads its
configuration from YAML, serves HTTP with rate limiting, and caches catalog
lookups. Billing lives in Python; the web client is TypeScript.

## Configuration

Settings are read from `deploy/config.yaml` and overridden by `SHOP_*`
environment variables.
