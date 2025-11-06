import tensorflow as tf
import platform

print("=" * 60)
print("📋 SISTEMA E BACKEND TENSORFLOW")
print("=" * 60)

# Versões e ambiente
print(f"Python: {platform.python_version()}")
print(f"TensorFlow: {tf.__version__}")
print(f"Compilado com CUDA: {tf.test.is_built_with_cuda()}")
print(f"Compilado com ROCm: {tf.test.is_built_with_rocm()}")

# Detectar dispositivos físicos
print("\n🔍 Dispositivos físicos detectados:")
for device in tf.config.list_physical_devices():
    print(" -", device)

# Teste rápido de operação com threads controladas
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("\n⚙️  Executando teste de multiplicação de matrizes...")
a = tf.random.normal([1000, 1000])
b = tf.random.normal([1000, 1000])
try:
    c = tf.matmul(a, b)
    print("✅ Operação concluída com sucesso:", c.shape)
except Exception as e:
    print("❌ Erro ao executar operação:", e)

# Teste de GPU (se existir)
if tf.config.list_physical_devices('GPU'):
    print("\n🧠 Teste com GPU disponível")
    with tf.device('/GPU:0'):
        try:
            c_gpu = tf.matmul(a, b)
            print("✅ GPU operando corretamente:", c_gpu.shape)
        except Exception as e:
            print("❌ Erro ao executar em GPU:", e)
else:
    print("\n⚠️ Nenhuma GPU detectada — execução apenas em CPU.")

print("=" * 60)
