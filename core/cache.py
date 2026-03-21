import hashlib

cache = {}


def get_cache_key(question: str):
    return hashlib.md5(question.encode()).hexdigest()


def get_cached(question: str):
    return cache.get(get_cache_key(question))


def set_cache(question: str, value):
    cache[get_cache_key(question)] = value