def truncate_reasoning(text, max_chars=200):
    if len(text) > max_chars:
        return text[:max_chars - 3] + "..."
    return text


def format_reasoning_for_csv(reasoning):
    reasoning = reasoning.replace('"', "'")
    return reasoning


def rank_consistency_check(results):
    for i in range(len(results) - 1):
        if results[i]["score"] < results[i + 1]["score"]:
            return False
    return True
