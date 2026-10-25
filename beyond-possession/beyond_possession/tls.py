"""Verified TLS, including Python.org macOS installs without a populated CA store."""
import ssl
import importlib

def verified_context():
    context=ssl.create_default_context()
    # Use the trusted public CA bundle shipped with certifi/pip, when available.
    # Hostname and certificate verification always remain enabled.
    for name in ['certifi','pip._vendor.certifi']:
        try:
            bundle=importlib.import_module(name).where()
            context.load_verify_locations(cafile=bundle)
            break
        except (ImportError,AttributeError,OSError,ssl.SSLError):
            continue
    return context
