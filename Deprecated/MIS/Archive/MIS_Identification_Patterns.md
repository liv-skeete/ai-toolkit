
        change_indicators: List[str] = Field(
            default=[
                r"(?:is|am|are) now",
                r"(?:is|am|are) currently",
                r"(?:has|have) (?:now|recently) changed to",
                r"(?:now|currently) (?:live|work|reside)s? in",
                r"(?:recently|just) moved to",
                r"(?:updated|changed) (?:my|his|her|their)? (.+?) to",
                r"(?:no longer|not anymore|stopped)",
                r"from now on",
                r"switched (?:from .+? )?to",
            ]

        attribute_patterns: List[str] = Field(
            default=[
                # Location attributes
                r"live(?:s)? in (.+)",
                r"(?:home|house|apartment) (?:is )?in (.+)",
                r"(?:reside|residing|resident) (?:in|of) (.+)",
                
                # Work/Professional attributes
                r"work(?:s)? (?:at|for) (.+)",
                r"(?:job|position|role|title) (?:is|as) (.+)",
                r"employed (?:at|by) (.+)",
                
                # Preference attributes
                r"favorite (.+?) is (.+)",
                r"prefer(?:s)? (.+?) over",
                r"like(?:s)? (?:to )?(.+)",
                
                # Relationship attributes
                r"(?:manager|boss|supervisor) is (.+)",
                r"(?:doctor|physician|therapist) is (.+)",
                r"(?:partner|spouse|husband|wife) is (.+)",
                
                # Contact attributes
                r"(?:phone|number|email|address) is (.+)",
                r"(?:live|stay|reside) at (.+)",
            ]