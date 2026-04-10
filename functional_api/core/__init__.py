"""Private implementation modules behind the public functional API.

The public entrypoints live in `functional_api`.
Code in this package exists so pipelines can share model-specific training,
data preparation, and benchmark logic without exposing those details as the
supported API surface.
"""
