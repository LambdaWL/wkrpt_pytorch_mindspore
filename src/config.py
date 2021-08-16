class Config:
    @staticmethod
    def schema():
        return {
            "num_class": {"type": int, "range_fn": lambda x: x > 0},
            "feature_dim": {"type": int, "range_fn": lambda x: x > 0},
            "hidden_dim": {"type": int, "range_fn": lambda x: x > 0},
            "dropout": {"type": float, "range_fn": lambda x: 1 >= x >= 0},
        }
    
    def raise_type_error(self, attr_str):
        raise ValueError(
            f"Type check failed for {attr_str}={getattr(self, attr_str)}. " 
            "Please check config.py for the correct type."
        )
    
    def raise_range_error(self, attr_str):
        raise ValueError(
            f"Range check failed for {attr_str}={getattr(self, attr_str)}. " 
            "Please check config.py for the correct range."
        )

    def __init__(self, num_class, feature_dim=1024, hidden_dim=300, dropout=0.0):
        self.num_class = num_class
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        for attr_str, schema in self.schema().items():
            attr = getattr(self, attr_str)
            if not isinstance(attr, schema["type"]):
                self.raise_type_error(attr_str)
            if not schema["range_fn"](attr):
                self.raise_range_error(attr_str)

        

