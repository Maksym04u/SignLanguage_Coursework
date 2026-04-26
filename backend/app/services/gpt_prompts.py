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
