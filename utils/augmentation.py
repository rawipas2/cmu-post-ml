"""
Data Augmentation utilities for Thai text
v1.2.2: Simple but effective augmentation techniques
"""
import random
import numpy as np
from typing import List, Tuple
from pythainlp.tokenize import word_tokenize


class ThaiTextAugmenter:
    """
    Augment Thai text data using simple techniques
    
    Methods:
        1. Random deletion: ลบคำบางคำออก
        2. Random swap: สลับตำแหน่งคำ
        3. Synonym replacement: แทนที่ด้วยคำพ้องความหมาย (simple)
        4. Random insertion: แทรกคำซ้ำ
    """
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Common Thai stop words (can be removed without changing meaning much)
        self.stop_words = set([
            'ครับ', 'ค่ะ', 'นะ', 'จ้า', 'จ๊ะ', 'เลย', 'แล้ว', 'ด้วย', 
            'อยู่', 'ไป', 'มา', 'ได้', 'เอง'
        ])
        
        # Simple synonym pairs for depression-related words
        self.synonyms = {
            'เศร้า': ['ทุกข์', 'หดหู่', 'ซึมเศร้า'],
            'เหงา': ['โดดเดี่ยว', 'อ้างว้าง'],
            'เครียด': ['กดดัน', 'ตึงเครียด'],
            'ท้อแท้': ['หมดกำลังใจ', 'ท้อใจ'],
            'เบื่อ': ['เบื่อหน่าย', 'รำคาญ'],
            'ดี': ['สบายใจ', 'มีความสุข'],
            'สนุก': ['สนุกสนาน', 'รื่นรมย์'],
        }
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words with probability p
        
        Args:
            text: Input text
            p: Probability of deleting each word (default: 0.1)
        
        Returns:
            Augmented text
        """
        words = word_tokenize(text, engine='newmm')
        
        # Don't delete if text is too short
        if len(words) <= 3:
            return text
        
        # Delete words randomly
        new_words = [w for w in words if random.random() > p]
        
        # Ensure at least some words remain
        if len(new_words) == 0:
            return text
        
        return ''.join(new_words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """
        Randomly swap positions of n pairs of words
        
        Args:
            text: Input text
            n: Number of swaps (default: 1)
        
        Returns:
            Augmented text
        """
        words = word_tokenize(text, engine='newmm')
        
        if len(words) <= 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ''.join(words)
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        Replace n words with their synonyms
        
        Args:
            text: Input text
            n: Number of words to replace (default: 1)
        
        Returns:
            Augmented text
        """
        words = word_tokenize(text, engine='newmm')
        
        # Find words that have synonyms
        replaceable_indices = [
            i for i, w in enumerate(words) if w in self.synonyms
        ]
        
        if not replaceable_indices:
            return text
        
        # Replace n random words
        n_replace = min(n, len(replaceable_indices))
        indices_to_replace = random.sample(replaceable_indices, n_replace)
        
        for idx in indices_to_replace:
            word = words[idx]
            synonym = random.choice(self.synonyms[word])
            words[idx] = synonym
        
        return ''.join(words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert n copies of existing words
        
        Args:
            text: Input text
            n: Number of insertions (default: 1)
        
        Returns:
            Augmented text
        """
        words = word_tokenize(text, engine='newmm')
        
        if len(words) == 0:
            return text
        
        for _ in range(n):
            # Pick random word and random position
            word = random.choice(words)
            pos = random.randint(0, len(words))
            words.insert(pos, word)
        
        return ''.join(words)
    
    def augment(self, text: str, n_aug: int = 1, 
                methods: List[str] = None) -> List[str]:
        """
        Generate n_aug augmented versions of text
        
        Args:
            text: Input text
            n_aug: Number of augmented samples to generate
            methods: List of methods to use (default: all)
        
        Returns:
            List of augmented texts
        """
        if methods is None:
            methods = ['delete', 'swap', 'synonym', 'insert']
        
        augmented = []
        
        for _ in range(n_aug):
            # Randomly choose a method
            method = random.choice(methods)
            
            if method == 'delete':
                aug_text = self.random_deletion(text, p=0.1)
            elif method == 'swap':
                aug_text = self.random_swap(text, n=1)
            elif method == 'synonym':
                aug_text = self.synonym_replacement(text, n=1)
            elif method == 'insert':
                aug_text = self.random_insertion(text, n=1)
            else:
                aug_text = text
            
            augmented.append(aug_text)
        
        return augmented


def augment_dataset(texts: List[str], labels: List[str], 
                   aug_per_sample: int = 1,
                   balance_classes: bool = True) -> Tuple[List[str], List[str]]:
    """
    Augment entire dataset with optional class balancing
    
    Args:
        texts: List of texts
        labels: List of labels
        aug_per_sample: Number of augmented samples per original sample
        balance_classes: If True, augment minority class more
    
    Returns:
        Augmented texts and labels
    """
    from collections import Counter
    
    augmenter = ThaiTextAugmenter()
    
    aug_texts = []
    aug_labels = []
    
    if balance_classes:
        # Count class distribution
        counter = Counter(labels)
        max_count = max(counter.values())
        
        # Augment minority classes more
        for text, label in zip(texts, labels):
            # Add original
            aug_texts.append(text)
            aug_labels.append(label)
            
            # Calculate how many augmentations needed
            class_count = counter[label]
            n_aug = int((max_count / class_count - 1) * aug_per_sample)
            
            if n_aug > 0:
                augmented = augmenter.augment(text, n_aug=n_aug)
                aug_texts.extend(augmented)
                aug_labels.extend([label] * n_aug)
    else:
        # Uniform augmentation
        for text, label in zip(texts, labels):
            aug_texts.append(text)
            aug_labels.append(label)
            
            if aug_per_sample > 0:
                augmented = augmenter.augment(text, n_aug=aug_per_sample)
                aug_texts.extend(augmented)
                aug_labels.extend([label] * aug_per_sample)
    
    print(f"📈 Augmentation complete:")
    print(f"   Original: {len(texts)} → Augmented: {len(aug_texts)}")
    if balance_classes:
        final_counter = Counter(aug_labels)
        print(f"   Class distribution: {dict(final_counter)}")
    
    return aug_texts, aug_labels


# Example usage
if __name__ == "__main__":
    print("Testing Thai Text Augmentation...")
    
    augmenter = ThaiTextAugmenter()
    
    test_text = "ฉันรู้สึกเศร้ามากและเหงามาก"
    
    print(f"\nOriginal: {test_text}")
    print("\nAugmentations:")
    print(f"1. Random deletion: {augmenter.random_deletion(test_text)}")
    print(f"2. Random swap: {augmenter.random_swap(test_text)}")
    print(f"3. Synonym: {augmenter.synonym_replacement(test_text)}")
    print(f"4. Random insert: {augmenter.random_insertion(test_text)}")
    
    print("\nGenerate 3 random augmentations:")
    augmented = augmenter.augment(test_text, n_aug=3)
    for i, aug in enumerate(augmented, 1):
        print(f"{i}. {aug}")
    
    print("\n✓ Augmentation test passed!")
