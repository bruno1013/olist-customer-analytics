"""
Text helpers for the Portuguese review comments (Phase 5).

normalize() lowercases and strips accents, so "Não recebi" and "nao recebi"
become the same text. The complaint THEMES are regular expressions written
against normalized text (no accents).
"""

import re
import unicodedata

import pandas as pd

# Portuguese stopwords: very common words with little meaning on their own.
# Negations ("nao", "nem", "nunca") and contrast words ("mas", "porem") are
# deliberately KEPT: "nao gostei" and "gostei" mean opposite things.
# "so" (só) and "ainda" are also kept: "so veio um", "ainda nao chegou".
# Written as one string for readability, hence the noqa for ruff's SIM905.
STOPWORDS = sorted(set("""
a o as os um uma uns umas de do da dos das no na nos nas em ao aos
por pelo pela pelos pelas para pra pro com sem sob sobre entre ate apos
e ou que se como quando onde quem qual quais cujo cuja
eu tu ele ela nos vos eles elas me te lhe lhes meu minha meus minhas
seu sua seus suas teu tua nosso nossa nossos nossas dele dela deles delas
este esta estes estas esse essa esses essas aquele aquela aqueles aquelas isto isso aquilo
foi era sao ser estar esta estou estava estao tem tinha ter ha havia
fui seja sido sendo vai vou ja tambem mais muito muita muitos muitas
la aqui ai ali entao assim bem sim
""".split()))  # noqa: SIM905

# Complaint themes, written against normalized text. Notes on the regex syntax:
#   \b      word boundary, so "atras" matches atraso / atrasou / atrasado
#   (?:...) a group that does NOT capture text (we only need True/False)
# Keywords are not sentiment: "dentro do prazo" is praise. So Delay and
# Poor quality need negative context, and themes are measured on bad reviews.
THEMES = {
    "Missing items / partial delivery": (
        r"\bfalt(?:a|ou|aram|ando|ava)\b|\bparcial|\bincomplet|\bum dos\b|\buma das\b"
        r"|\b(?:apenas|somente|so) (?:um|uma|1|o|a|parte)\b"
        r"|\b(?:so|apenas|somente) (?:veio|chegou|recebi)\b"
    ),
    "Not received": (
        r"\bnao (?:recebi|chegou|chegaram|veio|vieram|foi entregue|foram entregues)\b"
        r"|\bainda nao\b|\bnao receb|\bcade\b|\baguardando\b"
    ),
    "Delay": (
        r"\batras|\bdemor|\bfora do prazo\b|\bpassou do prazo\b|\bapos o prazo\b"
        r"|\bcorreios?\b|\bgreve\b"
    ),
    "Damaged / defective": (
        r"\bdefeit|\bquebrad|\bdanificad|\bamassad|\bestragad|\brasgad|\btrincad|\brachad"
        r"|\bnao funciona|\bparou de funcionar|\bvazad"
    ),
    "Wrong item": (
        r"\berrad|\bdiferente|\boutro produto\b|\bnao (?:e|era) o (?:que|produto)\b"
        r"|\bnao corresponde|\bnao condiz"
    ),
    "Poor quality": (
        r"\b(?:ma|baixa|pessima|horrivel|pouca) qualidade\b|\bqualidade (?:ruim|inferior|pessima|baixa)\b"
        r"|\bpessim|\bhorrivel|\bruim\b|\bfrageis?\b|\bacabamento\b"
    ),
    "Refund / customer service": (
        r"\breembols|\bdevolu|\bdevolv|\b[ae]storn|\bextorn|\bdinheiro de volta|\batendimento"
        r"|\bresposta\b|\bsem retorno\b|\bcontato\b|\breclama"
    ),
}


def normalize(text):
    """Lowercase, remove accents and punctuation, collapse spaces."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tag_themes(normalized):
    """Return a DataFrame with one True/False column per theme."""
    return pd.DataFrame(
        {theme: normalized.str.contains(pattern, regex=True) for theme, pattern in THEMES.items()},
        index=normalized.index,
    )
