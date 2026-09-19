"""Default edge/node visual constants — overridable via config.yaml render.*."""

EDGE_COLORS = {
    "nova":                "#F4A536",
    "adept_spoke":         "#E85D2C",
    "adept_graft":         "#9B7FE0",
    "semantic_inter":      "#3DDC97",
    "temporal":            "#4C8DFF",
    "temporal_influence":  "#F0349D",
}

EDGE_OPACITY = {
    "nova":                0.35,
    "adept_spoke":         0.5,
    "adept_graft":         0.4,
    "semantic_inter":      0.9,
    "temporal":            0.7,
    "temporal_influence":  0.7,
}

EDGE_WIDTH = {
    "nova":                2.0,
    "adept_spoke":         1.4,
    "adept_graft":         1.4,
    "semantic_inter":      1.5,
    "temporal":            1.5,
    "temporal_influence":  1.5,
}

EDGE_DASHED = set()

SIZE_MULT = {"mega": 1.9, "big": 1.3, "normal": 1.0}

NODE_ROLES = {
    "nova":       {"shape": "icosahedron"},
    "pioneer":    {"shape": "octahedron"},
    "hub":        {"shape": "diamond"},
    "latest":     {"shape": "sphere"},
    "member":     {"shape": "sphere"},
    "offspring":  {"shape": "sphere"},
    "orphan":     {"shape": "sphere_small"},
}
NODE_ROLES_DEFAULT = {"shape": "sphere"}

THREE_VERSION = "0.185.1"
