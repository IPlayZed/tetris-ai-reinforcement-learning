from huggingface_hub import HfApi

api = HfApi()

# Ezt át kell írni a saját felhasználónevedre és az általad választott repó nevére
# Pl.: "szterlcourse/my_agent"
repo_id = "IPlayZed/tetrisv2"

# Ide be kell írni a saját tokenedet, amit a Hugging Face oldalán tudsz létrehozni (https://huggingface.co/settings/token)
token = "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"

api.create_repo(
    repo_id="tetris",
    private=False,
    exist_ok=True,
    repo_type="model",
    token=token
)

api.upload_folder(
    folder_path="/",
    repo_id=repo_id,
    repo_type="model",
    token=token,
    ignore_patterns=["__*"]
)

api.upload_folder(
    folder_path="../models/",
    repo_id=repo_id,
    repo_type="model",
    token=token,
    ignore_patterns=["__*"]
)
