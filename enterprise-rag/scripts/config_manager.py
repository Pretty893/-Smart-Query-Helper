import os
import sys
import importlib
import threading
from typing import Dict, Any, Optional

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


class ConfigManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config_path: str = None):
        if self._initialized:
            return

        self._config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config_data.py"
        )
        self._config = {}
        self._observer = None
        self._load_config()
        self._start_watcher()
        self._initialized = True

    def _load_config(self):
        try:
            spec = importlib.util.spec_from_file_location("config_data", self._config_path)
            if spec and spec.loader:
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                
                self._config = {
                    name: getattr(config_module, name)
                    for name in dir(config_module)
                    if not name.startswith("_")
                }
                print(f"配置加载成功，共 {len(self._config)} 个配置项")
            else:
                print(f"无法加载配置文件: {self._config_path}")
        except Exception as e:
            print(f"配置加载失败: {e}")

    def _start_watcher(self):
        if not WATCHDOG_AVAILABLE:
            print("watchdog 未安装，跳过配置热更新监控")
            return

        class ConfigFileHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.src_path.endswith("config_data.py"):
                    print("检测到配置文件变更，正在重新加载...")
                    self._load_config()
                    print("配置已热更新")

        event_handler = ConfigFileHandler()
        self._observer = Observer()
        self._observer.schedule(
            event_handler,
            os.path.dirname(self._config_path),
            recursive=False
        )
        self._observer.start()
        print("配置热更新监控已启动")

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"ConfigManager has no attribute '{name}'")

    def stop(self):
        if self._observer:
            self._observer.stop()
            self._observer.join()


_config_manager = None


def get_config() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


if __name__ == "__main__":
    config = get_config()
    print(f"chat_model_name: {config.chat_model_name}")
    print(f"embedding_model_name: {config.embedding_model_name}")
