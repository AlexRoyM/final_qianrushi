import socket
import torch
from PIL import Image
import io
import torchvision.transforms as transforms
import resnet_18
import time
import matplotlib.pyplot as plt

# TCP服务器IP和端口
TCP_IP = '0.0.0.0'
TCP_PORT = 1234
# 定义相同的转换操作，与训练时相同
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 根据模型需要调整大小
    transforms.ToTensor()
])


class TCPServer:
    def __init__(self, ip, port, buffer_size=30000):
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.conn = None
        self.addr = None

    def start_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.ip, self.port))
        self.socket.listen(1)
        print("Waiting for a connection...")
        self.conn, self.addr = self.socket.accept()
        print(f'Connection established with: {self.addr}')

    def receive_exact_size(self, size):
        data = b""
        while len(data) < size:
            packet = self.conn.recv(min(size - len(data), self.buffer_size))
            if not packet:
                return None
            data += packet
        return data

    def close_connection(self):
        if self.conn:
            self.conn.close()
        if self.socket:
            self.socket.close()


def predict_image(image_data, model):
    image = Image.open(io.BytesIO(image_data))
    image = transform(image).unsqueeze(0)  # 增加一个批次维度
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不计算梯度
        outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item(), image


def show_image_with_prediction(image, prediction):
    # 转换张量格式以适应matplotlib
    image = image.squeeze(0)  # 移除批次维度
    image = image.permute(1, 2, 0)  # 改变通道顺序
    image = image.numpy()  # 转换为numpy数组

    plt.imshow(image)
    plt.title(f"Predicted: {prediction}")
    plt.show()


def save_image(image_data, file_name):
    with open(file_name, "wb") as img_file:
        img_file.write(image_data)


def main():
    # 加载模型
    model = resnet_18.GestureCNN()
    model.load_state_dict(torch.load('../model/resnet18.pth'))
    model.eval()  # 设置为评估模式

    tcp_server = TCPServer(TCP_IP, TCP_PORT)
    tcp_server.start_server()

    try:
        send_char = 0
        for i in range(50):
            image_size_data = tcp_server.receive_exact_size(4)
            if image_size_data is None:
                break
            image_size = int.from_bytes(image_size_data, byteorder='little')
            if image_size <= 0 or image_size > 30000:
                print(f"Invalid image size received: {image_size}")
                break

            image_data = tcp_server.receive_exact_size(image_size)
            if image_data is None:
                break
            # 保存图像数据
            # save_image(image_data, f'newdata/3/{i+100}.jpg')
            # print(f"Image {i} saved.")

            prediction, image = predict_image(image_data, model)
            print(f"Predicted: {prediction}")
            # show_image_with_prediction(image, prediction)

            tcp_server.conn.send(str(send_char).encode())
            print(f"{i} times Sent '{send_char}' to ESP32.")
            send_char = (send_char + 1) % 10

    finally:
        tcp_server.close_connection()


if __name__ == '__main__':
    main()
