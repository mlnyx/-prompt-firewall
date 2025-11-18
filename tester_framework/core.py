import pandas as pd
from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class Seed:
    """테스트할 개별 프롬프트(데이터)와 그 결과를 저장하는 데이터 클래스"""
    data: str
    label: str
    
    # 1단계 테스트 결과
    s1_decision: Optional[str] = None
    s1_rule_id: Optional[str] = None
    s1_message: Optional[str] = None
    
    # 2단계 테스트 결과
    s2_decision: Optional[str] = None
    s2_risk_score: Optional[float] = None
    
    # 에러 추적
    error: Optional[str] = None

    def __str__(self) -> str:
        return f"Seed(label={self.label}, data='{self.data[:30]}...', s1_decision={self.s1_decision}, s2_decision={self.s2_decision})"
    
    __repr__ = __str__

class Population:
    """Seed 객체의 컬렉션(테스트셋)을 관리하는 클래스"""
    def __init__(self, seeds: List[Seed] = None) -> None:
        self.seeds: List[Seed] = seeds if seeds else []

    def create_population_from_file(self, file_path: str):
        """
        제공된 CSV 파일 형식(text, label)에서 Population을 생성합니다.
        """
        try:
            df = pd.read_csv(file_path)
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("CSV must have 'text' and 'label' columns.")
                
            self.seeds = [
                Seed(data=str(row['text']), label=str(row['label']))
                for _, row in df.iterrows()
            ]
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
