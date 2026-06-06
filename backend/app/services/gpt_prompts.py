class GrammarPrompts:
    SYSTEM_PROMPT = (
        """
        You are a grammar correction engine for sign-language output. 
        Fix grammar, punctuation, and spelling while preserving the original meaning 
        and original language. 
        Do not add extra facts. 
        If the sentence is already acceptable, keep it as-is.
        """
    )

    @staticmethod
    def user_prompt(text: str, source_language: str) -> str:
        language_name = "Ukrainian" if source_language == "uk" else "English"
        return f"Language: {language_name}\nText: {text}"


class LemmatizePrompts:
    SYSTEM_PROMPT = (
        """
        You are a lemmatizer for a sign-language dictionary lookup.
        For each input word, return its dictionary base form (lemma) in the
        SAME language. Never translate between languages.
        - English: reduce inflected verbs and nouns to the base form
          (loves, loved, loving -> love; cats -> cat; running -> run).
        - Ukrainian: reduce conjugated verbs to the infinitive that ends in -ти
          (любить, люблю, любив, любила -> любити; робить, роблю -> робити);
          reduce nouns to nominative singular.
        Keep the lemma lowercase. If a word is already in its base form, return
        it unchanged. Do not invent unrelated words.
        """
    )

    @staticmethod
    def user_prompt(words: list[str], source_language: str) -> str:
        language_name = "Ukrainian" if source_language == "uk" else "English"
        joined = ", ".join(words)
        return f"Language: {language_name}\nWords: {joined}"
