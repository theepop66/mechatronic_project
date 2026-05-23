# 🚗 AI-Based Regenerative Suspension System
> A Quarter-Car prototype that uses Deep Reinforcement Learning to optimize the trade-off between **Ride Comfort** and **Energy Harvesting**.

## 📖 Project Overview
Traditional vehicle suspension systems waste kinetic energy as heat (friction) in oil dampers. This project replaces the standard damper with a **Regenerative damper**.

---

RegenerativeSuspension_Project/
│
├── 1_python_training/             # 🐍 เฟสที่ 1: สำหรับจำลอง Environment และเทรน AI
│   ├── envs/                      # โฟลเดอร์เก็บสภาพแวดล้อมจำลอง
│   │   └── quarter_car_env.py     # โค้ดสร้าง Custom Environment (สมการฟิสิกส์การสั่น, แรง Lorentz)
│   ├── agent/                     # โฟลเดอร์เก็บตัวโมเดล RL
│   │   └── rl_model.py            # สถาปัตยกรรม Neural Network (สร้างด้วย TensorFlow หรือ Keras)
│   ├── data/                      # เก็บไฟล์ Log หรือกราฟผลลัพธ์การลดการสั่นเทียบกับ PID
│   ├── train.py                   # สคริปต์หลักสำหรับกดรันเพื่อเริ่มเทรน RL Agent
│   ├── test_simulation.py         # สคริปต์สำหรับทดสอบรันโมเดลที่เทรนเสร็จแล้วในคอมพิวเตอร์
│   ├── export_to_c.py             # [สำคัญ!] สคริปต์แปลงโมเดลที่เทรนเสร็จเป็นไฟล์ .h (C-Array)
│   └── requirements.txt           # ไฟล์ระบุ Library ที่ต้องใช้ (เช่น tensorflow, gymnasium, numpy)
│
└── 2_esp32_firmware/              # ⚙️ เฟสที่ 2: โค้ด C++ สำหรับนำโมเดลไปฝังบน ESP32 (PlatformIO)
    ├── include/                   
    │   ├── Config.h               # ตั้งค่า Pin (เช่น ขาอ่าน Voltage, ขา I2C/SPI ของ Digital Pot)
    │   ├── RLController.h         # ประกาศคลาสรันโมเดล
    │   └── DigitalPot.h           # ประกาศคลาสคุม Hardware
    ├── src/                       
    │   ├── main.cpp               # Control Loop หลัก (รันทุกๆ X ms)
    │   ├── RLController.cpp       # โค้ดอ่านค่า Voltage เข้า Model และดึงค่า Action ออกมา
    │   └── DigitalPot.cpp         # โค้ดแปลงค่า Action เป็นสัญญาณส่งไปปรับความต้านทาน MOSFET
    ├── models/                    # 🧠 โฟลเดอร์เชื่อมต่อระหว่าง 2 เฟส
    │   └── model_data.h           # ไฟล์ที่ได้จาก export_to_c.py เอามาวางทับตรงนี้ได้เลย!
    └── platformio.ini             # ตั้งค่าบอร์ด ESP32 และความเร็วในการอัปโหลดโค้ด

By using a **Reinforcement Learning (RL) Agent**, the system actively controls the damping force in real-time. It stiffens the suspension to harvest maximum energy on highways and softens it instantly to absorb shocks on rough roads, ensuring passenger comfort while recharging the battery.

### Key Innovations
* **Regenerative Braking for Suspension:** Converts vertical vibration into electricity (12V DC).
* **AI Control Logic:** Replaces fixed PID/Passive damping with an adaptive RL Agent.
* **Physical Validation:** Tested on a custom-built Quarter-Car Test Rig.

---

## 🛠️ Hardware Architecture
The system is built on a "Quarter-Car" vertical slider rig.



---



## 💻 Software & Simulation (python & c++)


