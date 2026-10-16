"""Frozen mainline kernel extracted from evidence.render_input_contract."""

from __future__ import annotations


def validate_render_input(
    cache, *, prior_sha256, gaussian_type, sh_degree, background, allow_legacy=False
):
    source = cache.get("input", {})
    config = cache.get("configuration", {})
    if not source.get("gaussian_ply_sha256"):
        if not allow_legacy:
            raise ValueError(
                "render cache lacks a prior SHA binding; regenerate it or explicitly opt into legacy unbound audit"
            )
        verified = False
    else:
        if source["gaussian_ply_sha256"] != prior_sha256:
            raise ValueError(
                "render audit Gaussian prior differs from the cached observations"
            )
        verified = True
    for key, value in [
        ("gaussian_type", gaussian_type),
        ("sh_degree", int(sh_degree)),
        ("background", background),
    ]:
        if config.get(key) != value:
            raise ValueError(f"render audit {key} differs from the cached observations")
    return verified
