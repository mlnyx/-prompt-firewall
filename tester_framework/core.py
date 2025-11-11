import pandas as pd
from typing import List

class Seed:
    """테스트할 개별 프롬프트(데이터)와 그 결과를 저장하는 클래스"""
    def __init__(self, data: str, label: str):
        self.data = data
        self.label = label  # Ground truth (e.g., 'benign', 'jailbreak')
        
        # 1단계 테스트 결과 
        self.s1_decision = None
        self.s1_rule_id = None      
        self.s1_message = None      
        
        # 2단계 테스트 결과
        self.s2_decision = None
        self.s2_risk_score = None
        
        # error 추적
        self.error = None


    def __str__(self) -> str:
        return f"Seed(label={self.label}, data='{self.data[:30]}...', api_decision={self.api_decision})"
    
    __repr__ = __str__

class Population:
    """Seed 객체의 컬렉션(테스트셋)을 관리하는 클래스"""
    def __init__(self, seeds: List[Seed] = None) -> None:
        self.seeds = seeds if seeds else []

    def create_population_from_file(self, file_path: str):
        """
        제공된 CSV 파일 형식(text, label)에서 Population을 생성합니다.
        """
        try:
            df = pd.read_csv(file_path)
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("CSV must have 'text' and 'label' columns.")
                
            for _, row in df.iterrows():
                self.seeds.append(Seed(data=str(row['text']), label=str(row['label'])))
            print(f"Successfully loaded {len(self.seeds)} seeds from {file_path}")
        
        except FileNotFoundError:
            print(f"Error: Data file not found at {file_path}")
            self.seeds = []
        except Exception as e:
            print(f"Error loading population: {e}")
            self.seeds = []

    def __len__(self):
        return len(self.seeds)

    def __iter__(self):
        return iter(self.seeds)

    

    

    

    

    


