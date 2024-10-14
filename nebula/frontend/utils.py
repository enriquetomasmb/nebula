class Utils:
    def __init__(self):
        self.init()
        
    @classmethod
    def check_path(cls, base_path, full_path):
        if not full_path.startswith(base_path):
            raise Exception("Not allowed")
        return full_path
        