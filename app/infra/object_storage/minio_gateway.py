from minio import Minio
from app.shared.clients.minio_utils import get_minio_client
from app.infra.config.providers import infra_config


class MinioGateway:
    """
    MinIO 对象存储网关类
    统一封装 MinIO 客户端获取、配置读取、图片URL拼接等能力，
    供全项目统一调用，避免到处写配置、重复拼接URL
    """
    # 桶
    @property
    def bucket_name(self) -> str:
        return infra_config.minio_config.bucket_name


    # 图片文件夹前缀
    @property
    def image_dir(self) -> str:
        return infra_config.minio_config.minio_img_dir


    # 客户端对象
    @property
    def client(self) -> Minio:
        return get_minio_client()


    # 构建访问地址（minio_image = 端点+桶+对象名）
    def build_image_url(self, stem:str, image_name:str) -> str:
        protocol = "https" if infra_config.minio_config.minio_secure else "http"
        # 拼接地址
        return (
            f"{protocol}://{infra_config.minio_config.endpoint}/"
            f"{self.bucket_name}{self.image_dir}/{stem}/{image_name}"
        )

minio_gateway = MinioGateway()


