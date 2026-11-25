"""Versioned bounded Hamming recovery and compact-tag research controls.

No image/model/source-answer access. SHA256 metadata is consistency,not authenticity.
"""

import hashlib
import hmac
import itertools
from . import integrity as g

TAG_BITS = (16, 24, 32, 64, 256)
TAG_DOMAIN = b"BrAIn-visual-recovery-v2\x00"


def candidates(values, radius):
    if type(radius) is not int or radius not in (0, 1, 2):
        raise ValueError("unsupported_radius")
    payload = g.pack_values(values)
    for distance in range(1, radius + 1):
        for positions in itertools.combinations(range(16), distance):
            alternatives = [[v for v in range(4) if v != values[p]] for p in positions]
            for replacements in itertools.product(*alternatives):
                packet = bytearray(payload)
                for p, v in zip(positions, replacements):
                    packet[p // 4] ^= (values[p] ^ v) << (6 - 2 * (p % 4))
                yield bytes(packet), positions


def search(values, radius, context, expected):
    matches = []
    checks = 0
    for packet, positions in candidates(values, radius):
        digest = context.copy()
        digest.update(packet)
        checks += 1
        if hmac.compare_digest(digest.digest()[: len(expected)], expected):
            matches.append((packet, positions))
    if len(matches) != 1:
        return {
            **g.unavailable("no_unique_bounded_match"),
            "recovered": False,
            "distance": None,
            "candidate_checks": checks,
            "matches": len(matches),
        }
    packet, positions = matches[0]
    return {
        "status": "accepted",
        "values": g.unpack_values(packet),
        "recovered": True,
        "distance": len(positions),
        "changed_positions": list(positions),
        "candidate_checks": checks,
        "matches": 1,
    }


def recover(values, manifest, profile, radius=2):
    if type(radius) is not int or radius not in (0, 1, 2):
        return {
            **g.unavailable("unsupported_radius"),
            "recovered": False,
            "distance": None,
            "candidate_checks": 0,
        }
    receipt = g.verify_readout(values, manifest, profile)
    if receipt["status"] == "accepted":
        return {**receipt, "recovered": False, "distance": 0, "candidate_checks": 0}
    if receipt["reason"] != "checksum_mismatch" or radius == 0:
        return {**receipt, "recovered": False, "distance": None, "candidate_checks": 0}
    result = search(
        values,
        radius,
        hashlib.sha256(g.DOMAIN + bytes.fromhex(g.profile_id(profile))),
        bytes.fromhex(manifest["payload_sha256"]),
    )
    if result["status"] == "accepted":
        result["profile_sha256"] = g.profile_id(profile)
    return result


def make_tag(values, profile, bits):
    if type(bits) is not int or bits not in TAG_BITS:
        raise ValueError("unsupported_tag_bits")
    pid = g.profile_id(profile)
    digest = hashlib.sha256(
        TAG_DOMAIN + bits.to_bytes(2, "big") + bytes.fromhex(pid) + g.pack_values(values)
    ).digest()[: bits // 8]
    return {"version": 2, "bits": bits, "profile_sha256": pid, "tag": digest.hex()}


def recover_tag(values, manifest, profile, radius=2):
    try:
        if type(radius) is not int or radius not in (0, 1, 2):
            raise ValueError("unsupported_radius")
        pid = g.profile_id(profile)
        payload = g.pack_values(values)
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"version", "bits", "profile_sha256", "tag"}
            or type(manifest["version"]) is not int
            or manifest["version"] != 2
            or type(manifest["bits"]) is not int
            or manifest["bits"] not in TAG_BITS
        ):
            raise ValueError("invalid_tag_manifest")
        bits = manifest["bits"]
        tag = manifest["tag"]
        if (
            manifest["profile_sha256"] != pid
            or not isinstance(tag, str)
            or len(tag) != bits // 4
            or any(c not in "0123456789abcdef" for c in tag)
        ):
            raise ValueError("invalid_tag_manifest")
        expected = bytes.fromhex(tag)
        context = hashlib.sha256(TAG_DOMAIN + bits.to_bytes(2, "big") + bytes.fromhex(pid))
        digest = context.copy()
        digest.update(payload)
        if hmac.compare_digest(digest.digest()[: bits // 8], expected):
            return {
                "status": "accepted",
                "values": list(values),
                "recovered": False,
                "distance": 0,
                "candidate_checks": 0,
                "profile_sha256": pid,
            }
        if radius == 0:
            return {
                **g.unavailable("checksum_mismatch"),
                "recovered": False,
                "distance": None,
                "candidate_checks": 0,
            }
        result = search(values, radius, context, expected)
        if result["status"] == "accepted":
            result["profile_sha256"] = pid
        return result
    except (ValueError, TypeError, KeyError):
        return {
            **g.unavailable("invalid_tag_input"),
            "recovered": False,
            "distance": None,
            "candidate_checks": 0,
        }
