"""Explicit opt-in for the connected 0..3 prefix, separate from old gates."""


def validate_layer(layer, registered, profile="qualified"):
    if (type(layer) is not int or profile not in ("qualified", "token_prefix")
            or layer not in registered and not (profile == "token_prefix" and layer == 1)):
        raise ValueError("Unregistered original layer/profile; token_prefix only adds layer 1")
    return layer
