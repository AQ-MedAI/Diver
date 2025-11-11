# 环境配置（中文版）
## 1. 使用 pip 自动安装依赖库

```bash
cd env_requirements
pip install -r requirements.txt
```

## 2. 需要手动安装的库
### 2.1 安装 pytrec_eval - 搜索评估工具库

报错信息：
```
ModuleNotFoundError: No module named 'pytrec_eval'
```

直接使用 `pip install pytrec_eval` 可能无法安装。解决步骤如下：

```bash
# 参考：https://zhuanlan.zhihu.com/p/715975263
# 确保在 env_requirements 目录下操作，相关压缩包已下载至此
cd env_requirements

# 解压预下载的 pytrec_eval 和 trec_eval 压缩包，注意目标路径
unzip pytrec_eval-master.zip
tar -xzf trec_eval-9.0.8.tar.gz -C pytrec_eval-master

# 重命名，以符合setup.py预设的文件名
cd pytrec_eval-master/ && mv trec_eval-9.0.8 trec_eval

# 安装 pytrec_eval
python setup.py install
```

### 2.2 安装 JDK 21 - Java开发工具包

报错信息：
```
SystemError: JVM failed to start: -1
```

原因： pyserini 底层依赖 Java 编写的 Lucene 搜索引擎库，需要正确配置 JVM 环境。解决步骤如下：

```bash
# 参考：https://juejin.cn/post/7316202808984780827
# 下载 JDK 21 安装包
cd env_requirements
wget https://download.oracle.com/java/21/archive/jdk-21.0.7_linux-x64_bin.tar.gz  # 较大，190M

# 创建安装目录并解压
mkdir -p /opt/jdk
tar -xzf jdk-21.0.7_linux-x64_bin.tar.gz -C /opt/jdk/

# 配置环境变量
echo 'export JAVA_HOME=/opt/jdk/jdk-21.0.7' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc

# 使配置生效
source ~/.bashrc

# 验证安装
java --version
```

预期输出：
```
java 21.0.7 2025-04-15 LTS
Java(TM) SE Runtime Environment (build 21.0.7+8-LTS-245)
Java HotSpot(TM) 64-Bit Server VM (build 21.0.7+8-LTS-245, mixed mode, sharing)
```

---

# Environment Setup (English Version)

## 1. Automatic Installation via pip
```bash
cd env_requirements
pip install -r requirements.txt
```

## 2. Manual Installation Requirements

### 2.1 Install pytrec_eval Library

Error Message:
```
ModuleNotFoundError: No module named 'pytrec_eval'
```

Solution:

Direct `pip install pytrec_eval` may fail. Please follow these steps:

```bash
# Ensure you are in the env_requirements directory, where the required packages have been downloaded
cd env_requirements

# Extract the pre-downloaded pytrec_eval and trec_eval packages, pay attention to the target path
unzip pytrec_eval-master.zip
tar -xzf trec_eval-9.0.8.tar.gz -C pytrec_eval-master

# Rename to match the filename expected by setup.py
cd pytrec_eval-master/ && mv trec_eval-9.0.8 trec_eval

# Install pytrec_eval
python setup.py install
```

### 2.2 Install JDK 21

Error Message:
```
SystemError: JVM failed to start: -1
```

Reason: pyserini relies on the Lucene search engine library written in Java, which requires proper JVM environment configuration. Follow these steps to resolve:

Installation Steps:

```bash
# Download JDK 21 installation package
cd env_requirements
wget https://download.oracle.com/java/21/archive/jdk-21.0.7_linux-x64_bin.tar.gz

# Create installation directory and extract
mkdir -p /opt/jdk
tar -xzf jdk-21.0.7_linux-x64_bin.tar.gz -C /opt/jdk/

# Configure environment variables
echo 'export JAVA_HOME=/opt/jdk/jdk-21.0.7' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc

# Apply the configuration
source ~/.bashrc

# Verify installation
java --version
```

Expected Output:
```
java 21.0.7 2025-04-15 LTS
Java(TM) SE Runtime Environment (build 21.0.7+8-LTS-245)
Java HotSpot(TM) 64-Bit Server VM (build 21.0.7+8-LTS-245, mixed mode, sharing)
```