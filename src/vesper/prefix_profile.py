"""Explicit opt-in for the connected 0..3 prefix, separate from old gates."""


def validate_layer(layer, registered, profile="qualified"):
    if (type(layer) is not int or profile not in ("qualified", "token_prefix", "backbone_probe")
            or layer not in registered and not (profile == "token_prefix" and layer == 1)
            and not (profile == "backbone_probe" and 0 <= layer < 40)):
        raise ValueError("Unregistered original layer/profile")
    return layer
