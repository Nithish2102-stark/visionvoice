"""
Hierarchical OCR Text Reconstructor.
Rebuilds OCR output structured by block -> paragraph -> line -> words
to preserve original book typography, paragraphs, line breaks, and punctuation.
"""

from __future__ import annotations
from typing import Dict, List, Any, Tuple
from visionvoice.utils.logging import get_logger

logger = get_logger("TextReconstructor")


class TextReconstructor:
    """
    Reconstructs structured text from pytesseract.image_to_data() dictionary.
    Avoids flat word concatenation and preserves line/paragraph hierarchy.
    """

    def reconstruct(self, data_dict: Dict[str, List[Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Groups recognized words hierarchically by (block_num, par_num, line_num).
        Returns:
            (structured_text, filtered_words_data)
        """
        if not data_dict or "text" not in data_dict:
            return "", []

        n_items = len(data_dict["text"])
        if n_items == 0:
            return "", []

        # Hierarchical structure:
        # blocks: Dict[block_num, Dict[par_num, Dict[line_num, List[Tuple[word_text, word_dict]]]]]
        blocks: Dict[int, Dict[int, Dict[int, List[Tuple[str, Dict[str, Any]]]]]] = {}
        all_valid_words: List[Dict[str, Any]] = []

        for i in range(n_items):
            raw_word = str(data_dict["text"][i]).strip()
            conf = float(data_dict["conf"][i])

            if not raw_word:
                continue

            block_num = int(data_dict["block_num"][i])
            par_num = int(data_dict["par_num"][i])
            line_num = int(data_dict["line_num"][i])
            word_num = int(data_dict["word_num"][i])

            word_meta = {
                "text": raw_word,
                "conf": conf,
                "block_num": block_num,
                "par_num": par_num,
                "line_num": line_num,
                "word_num": word_num,
                "left": int(data_dict["left"][i]),
                "top": int(data_dict["top"][i]),
                "width": int(data_dict["width"][i]),
                "height": int(data_dict["height"][i]),
            }
            all_valid_words.append(word_meta)

            if block_num not in blocks:
                blocks[block_num] = {}
            if par_num not in blocks[block_num]:
                blocks[block_num][par_num] = {}
            if line_num not in blocks[block_num][par_num]:
                blocks[block_num][par_num][line_num] = []

            blocks[block_num][par_num][line_num].append((raw_word, word_meta))

        # Reconstruct hierarchical text: paragraphs separated by \n\n, lines separated by \n
        paragraph_texts: List[str] = []

        for block_idx in sorted(blocks.keys()):
            for par_idx in sorted(blocks[block_idx].keys()):
                line_texts: List[str] = []
                for line_idx in sorted(blocks[block_idx][par_idx].keys()):
                    words_in_line = [w[0] for w in blocks[block_idx][par_idx][line_idx]]
                    line_str = " ".join(words_in_line).strip()
                    if line_str:
                        line_texts.append(line_str)
                
                if line_texts:
                    par_str = "\n".join(line_texts).strip()
                    if par_str:
                        paragraph_texts.append(par_str)

        structured_text = "\n\n".join(paragraph_texts).strip()
        return structured_text, all_valid_words
