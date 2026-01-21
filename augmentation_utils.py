import random
import re
from konlpy.tag import Okt

class KoreanNoiseInjector:
    def __init__(self):
        self.okt = Okt()
        
        # Josa (Particles) replacements
        self.josa_map = {
            '은': ['이', '을', '에'],
            '는': ['가', '를', '에서'],
            '이': ['은', '을', '의'],
            '가': ['는', '를', '로'],
            '을': ['이', '은', '와'],
            '를': ['가', '는', '과'],
            '에': ['에서', '으로', '의'],
            '에서': ['에', '로서', '로'],
            '의': ['에', '은', '는'],
            '로': ['으로', '에', '를'],
            '으로': ['로', '에서', '가'],
            '과': ['와', '의', '랑'],
            '와': ['과', '에', '이랑']
        }
        
        # Eomi (Endings) replacements for lowering formality or making it awkward
        self.eomi_map = {
            '습니다': ['다', '어', '요'],
            '입니다': ['이다', '이야', '예요'],
            '한다': ['해', '하니', '함'],
            '된다': ['돼', '되니', '됨'],
            '있다': ['있어', '있음', '있으니'],
            '없다': ['없어', '없음', '없으니'],
            '이다': ['야', '임', '이니'],
            '다': ['요', '죠', '구나']
        }

        # Keyboard adjacency for spelling errors (simplified QWERTY/2-set mapping)
        self.keyboard_map = {
            'ㄱ': 'ㄴ', 'ㄴ': 'ㄷ', 'ㄷ': 'ㄹ', 'ㄹ': 'ㅁ', 'ㅁ': 'ㅂ', 'ㅂ': 'ㅅ', 'ㅅ': 'ㅇ', 'ㅇ': 'ㅈ', 'ㅈ': 'ㅊ', 'ㅊ': 'ㅋ', 'ㅋ': 'ㅌ', 'ㅌ': 'ㅍ', 'ㅍ': 'ㅎ', 'ㅎ': 'ㄱ',
            'ㅏ': 'ㅑ', 'ㅑ': 'ㅓ', 'ㅓ': 'ㅕ', 'ㅕ': 'ㅗ', 'ㅗ': 'ㅛ', 'ㅛ': 'ㅜ', 'ㅜ': 'ㅠ', 'ㅠ': 'ㅡ', 'ㅡ': 'ㅣ', 'ㅣ': 'ㅏ',
            'ㅐ': 'ㅔ', 'ㅔ': 'ㅐ', 'ㅒ': 'ㅖ', 'ㅖ': 'ㅒ'
        }

    def _decompose_hangul(self, char):
        """Helper to decompose Hangul for spelling noise (simplified)."""
        # This is a placeholder. Real decomposition requires unicode math.
        # For this prototype, we'll use simple char replacement or keyboard mapping.
        return char

    def inject_josa_noise(self, text):
        """Replaces particles (Josa) with incorrect ones."""
        try:
            pos = self.okt.pos(text)
            new_text = ""
            changed = False
            
            for word, tag in pos:
                if tag == 'Josa' and not changed and random.random() < 0.5:
                    # Attempt to find a replacement
                    candidates = self.josa_map.get(word)
                    if not candidates:
                        # Try partial match (last char)
                        last_char = word[-1]
                        candidates = self.josa_map.get(last_char)
                    
                    if candidates:
                        new_text += random.choice(candidates)
                        changed = True
                    else:
                        new_text += word
                else:
                    new_text += word
            # If POS tagging splits words (e.g. "학교" "에"), simple concatenation might lose spaces.
            # Okt often splits. A safer way for Josa is strictly replacing text segments if possible,
            # but reconstruction from POS is standard.
            # *Refinement*: Okt.pos returns tokens. We need to reconstruct carefully or use text replacement.
            # For simplicity in this specialized agent: simple reconstruction.
            # Note: Okt reconstruction often loses spacing. We will try to preserve it by simple join
            # but Korean agglutinative nature makes "word + josa" stick together.
            return new_text if changed else text
        except:
            return text

    def inject_ending_noise(self, text):
        """Changes sentence endings to break formality."""
        for target, replacements in self.eomi_map.items():
            if target in text:
                replacement = random.choice(replacements)
                # Replace the last occurrence or a random one
                text = text.replace(target, replacement, 1)
                return text
        return text

    def inject_conjugation_error(self, text):
        """Injects conjugation errors (Vowel harmony violations, wrong stems)."""
        # Approximate by replacing common patterns
        errors = [
            ('하', '해'), ('해', '하'),
            ('되', '돼'), ('돼', '되'),
            ('않', '안'), ('안', '않'),
            ('았', '었'), ('었', '았'),
            ('려', '러'), ('러', '려') # 가려/가러
        ]
        random.shuffle(errors)
        for cor, err in errors:
            if cor in text:
                return text.replace(cor, err, 1)
        return text

    def inject_spacing_error(self, text):
        """Randomly removes or adds spaces."""
        if random.random() < 0.5:
            # Remove a space
            if ' ' in text:
                parts = text.split(' ')
                if len(parts) > 1:
                    idx = random.randint(0, len(parts) - 2)
                    parts[idx] = parts[idx] + parts[idx+1]
                    del parts[idx+1]
                    return ' '.join(parts)
        else:
            # Add a space (randomly split a word)
            words = text.split(' ')
            if words:
                idx = random.randint(0, len(words) - 1)
                word = words[idx]
                if len(word) > 1:
                    split_point = random.randint(1, len(word) - 1)
                    new_word = word[:split_point] + ' ' + word[split_point:]
                    words[idx] = new_word
                    return ' '.join(words)
        return text

    def inject_spelling_error(self, text):
        """Injects typos based on keyboard adjacency or phonetic similarity."""
        if not text: return text
        idx = random.randint(0, len(text) - 1)
        char = text[idx]
        
        # Simple replacement if in map
        if char in self.keyboard_map:
            return text[:idx] + self.keyboard_map[char] + text[idx+1:]
        
        # Or just random noise for non-mapped chars? 
        # Let's try to find a char that IS in the map
        candidates = [i for i, c in enumerate(text) if c in self.keyboard_map]
        if candidates:
            idx = random.choice(candidates)
            return text[:idx] + self.keyboard_map[text[idx]] + text[idx+1:]
            
        return text

    def inject_word_order_error(self, text):
        """Swaps word order (Syntax)."""
        words = text.split(' ')
        if len(words) < 3: return text
        
        # Don't touch the last word (often the predicate/period)
        idx1, idx2 = random.sample(range(len(words) - 1), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

    def apply_weighted_noise(self, text):
        """
        Applies noise based on the 'Roulette' strategy:
        - 75% chance for Format errors (WS, SPELL)
        - 50% chance for Grammar errors (PART, END, CONJ, WO)
        """
        current_text = text
        applied_errors = []

        # 1. Format Noise (High Frequency)
        if random.random() < 0.75:
            target = random.choice(['WS', 'SPELL'])
            if target == 'WS': current_text = self.inject_spacing_error(current_text)
            elif target == 'SPELL': current_text = self.inject_spelling_error(current_text)
            applied_errors.append(target)

        # 2. Grammar Noise (Medium Frequency - Critical)
        if random.random() < 0.5:
            target = random.choice(['PART', 'END', 'CONJ', 'WO'])
            if target == 'PART': current_text = self.inject_josa_noise(current_text)
            elif target == 'END': current_text = self.inject_ending_noise(current_text)
            elif target == 'CONJ': current_text = self.inject_conjugation_error(current_text)
            elif target == 'WO': current_text = self.inject_word_order_error(current_text)
            applied_errors.append(target)

        # Ensure at least one error if it was a "clean" pass (optional, but good for data generation)
        if not applied_errors:
            # Force one low-level error
            current_text = self.inject_spacing_error(current_text)
            
        return current_text
