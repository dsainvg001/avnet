IO-VNBD: Inertial Odometry Vehicle Navigation Benchmark Dataset Guide
1. Overview & Dataset Purpose
The Inertial Odometry Vehicle Navigation Benchmark Dataset (IO-VNBD) is a public, large-scale, information-rich dataset designed for developing, benchmarking, and evaluating vehicle positioning, dead-reckoning odometry, and deep learning-based inertial navigation algorithms.

Recorded on public roads across the United Kingdom, Nigeria, and France, the dataset captures diverse driving conditions (country roads, motorways, urban traffic, roundabouts) and dynamic maneuvers (hard braking, sharp turns, acceleration).

Core Statistics
Vehicle CAN Bus / Ego-Motion Data (V-Dataset): ~40 hours driving time over 1,300+ km.
Smartphone Sensor Data (S-Dataset): ~58 hours driving time over 4,400+ km.
Sampling Rate: ~10 Hz (0.1 s sample period) across ego-motion sensors and smartphone sensors.
2. Directory Structure & File Categorization
The dataset is organized into two primary top-level folders:

Synchronised V abd S datasets/: Time-synchronized recordings where vehicle CAN bus data and Android smartphone sensor data share aligned time windows.
Unsynchronised V and S Dataset/: Raw, independent streams of vehicle CAN bus data and smartphone data.
Inside each primary directory, data is grouped into:

Categorised IOVNB Dataset: Grouped by driver profile and maneuver type (e.g., Driver A, Driver C, Driver E).
Uncategorised IOVNB Dataset: Direct recordings separated into:
V-Dataset/: Vehicle CAN bus files named with prefix V- (e.g., V-S2.csv).
S-Dataset/: Smartphone sensor files named with prefix S- (e.g., S-S2.csv).
3. Data Schema & Feature Specifications
3.1 Vehicle CAN Bus Dataset (V-Dataset)
The V-Dataset contains 29 columns extracted directly from vehicle ego-motion sensors and the CAN bus.

Column Name	Units	Range / Format	Description
No of GPS Satellites Available	Count	0 – 32	Number of visible GPS satellites.
Time Since Start of Day (seconds)	Seconds	0 – 86400	Seconds elapsed since 00:00 UTC.
Latitude (degrees)	Degrees	Float	WGS84 GPS Latitude position.
Longitude (degrees)	Degrees	Float	WGS84 GPS Longitude position.
Velocity (km/hr)	km/h	Float	Ground speed measured by GPS.
Heading (degrees)	Degrees	0.0° – 360.0°	GPS True North orientation angle.
Height (km)	km	Float	Altitude above sea level in kilometers.
Vertical velocity (km/hr)	km/h	Float	Vertical climb/descent speed.
Sample period (seconds)	Seconds	~0.1	Time interval between consecutive samples (10 Hz).
Steering Angle (degrees)	Degrees	Float	Hand-wheel steering angle.
Wheel Speed Front Left (rad/sec)	rad/s	Float	Rotational speed of Front Left wheel.
Wheel Speed Front Right (rad/sec)	rad/s	Float	Rotational speed of Front Right wheel.
Wheel Speed Rear Left (rad/sec)	rad/s	Float	Rotational speed of Rear Left wheel.
Wheel Speed Rear Right (rad/sec)	rad/s	Float	Rotational speed of Rear Right wheel.
Yaw Rate (deg/sec)	deg/s	Float	Vehicle angular turn rate around Z-axis.
Indicated Vehicle Speed (km/hr)	km/h	Float	Vehicle dashboard/wheel-speed integrated speed.
Indicated Longitudinal Acceleration (g)	g ($9.81 m/s^2$)	Float	Forward acceleration/deceleration.
Indicated Lateral Acceleration (g)	g ($9.81 m/s^2$)	Float	Side-to-side centripetal acceleration.
Handbrake (0 or 1)	Boolean	0 or 1	Handbrake engagement flag.
Gear Requested (1-5)	Integer	0 – 5	Gear position requested by driver.
Gear (1-5)	Integer	0 – 5	Currently engaged mechanical gear.
Engine Speed (rev/min)	RPM	Integer	Engine crankshaft RPM.
Coolant Temperature (degrees)	°C	Float	Engine coolant temperature.
Clutch Position (0 or 1)	Boolean	0 or 1	Clutch pedal pressed status.
Brake Pressure (psi)	PSI	Float	Hydraulic brake line fluid pressure.
Brake Position (0 or 1)	Boolean	0 or 1	Brake pedal pressed status flag.
Battery Voltage (volts)	Volts	Float	Electrical system supply voltage.
Air Temperature (degrees)	°C	Float	Ambient intake air temperature.
Accelerator Pedal Position (0 or 1)	Boolean	0 or 1	Throttle pedal engagement flag.
3.2 Smartphone Sensor Dataset (S-Dataset)
The S-Dataset contains 24 columns captured via an Android smartphone attached inside the vehicle sampling at 10 Hz.

Column Name	Units	Description
GPS LATITUDE (degrees)	Degrees	Smartphone GPS Latitude position.
GPS LONGITUDE (degrees)	Degrees	Smartphone GPS Longitude position.
GPS ALTITUDE (m)	Meters	Smartphone GPS Altitude.
GPS SPEED (Kmh)	km/h	Smartphone GPS Ground speed.
GPS ACCURACY (m)	Meters	Horizontal position accuracy estimate.
GPS ORIENTATION (°)	Degrees	Smartphone GPS Heading direction.
GPS SATELLITES IN RANGE	Ratio	Connected vs visible satellites (e.g. 29 / 29).
TIME SINCE START (ms)	Milliseconds	Time elapsed since recording start.
DATE (YYYY-MO-DD HH-MI-SS_SSS)	Timestamp	Date and UTC time string.
ACCELEROMETER X / Y / Z (m/s²)	$m/s^2$	Raw 3-axis linear acceleration (includes gravity).
GRAVITY X / Y / Z (m/s²)	$m/s^2$	Isolated 3-axis gravity vector components.
GYROSCOPE X / Y / Z (rad/s)	rad/s	Raw 3-axis angular rates of phone body.
MAGNETIC FIELD X / Y / Z (μT)	$\mu T$	3-axis ambient magnetic field strength.
ORIENTATION (Azimuth/Pitch/Roll) (°)	Degrees	Fused phone orientation Euler angles.
4. Odometry vs. GPS Dynamics & Correlation Findings
4.1 Speed & Cross-Correlation Characteristics
Pearson Correlation: High correlation ($\ge 0.999$) between Indicated vehicle speed and GPS Velocity under cruising conditions.
Signal Lag: SciPy cross-correlation analysis (scipy.signal.correlate) reveals a slight lag (~0.1 s / 1 sample) due to GPS Doppler filter smoothing relative to immediate wheel CAN bus speed readings.
4.2 Impact of Driving Behavior
Normal Driving:
Wheel speeds across all 4 wheels remain highly uniform (left/right difference $< 0.02$ rad/s).
Dead reckoning integrated over short-to-medium intervals ($< 1$ km) aligns closely with GPS ground truth.
Aggressive Driving (Hard Braking & Sharp Turning):
Sharp Turns / Roundabouts: High Yaw Rate ($> 15^\circ/s$) causes significant differentials between inner and outer wheel speeds ($> 0.24$ rad/s). Outer wheels rotate faster during tight cornering.
Hard Braking: Elevated Brake Pressure ($> 200$ PSI) causes mild wheel slip / tyre compression, introducing integration drift in pure wheel-speed dead reckoning.
Dead-Reckoning Drift: Uncorrected yaw rate sensor bias leads to quadratic position error accumulation over longer trajectories ($> 5$ km), highlighting the requirement for sensor fusion or neural error correction models.
5. Usage & Analysis Code
A complete script analyze_iovnbd.py is included in the root directory. It executes:

Signal zero-mean cross-correlation and optimal time lag calculation using SciPy.
Conversion of WGS84 GPS Lat/Lon to local East-North-Up (ENU) meter coordinates.
Dead-reckoning 2D trajectory reconstruction from vehicle speed and integrated yaw rate.
Matplotlib figure generation comparing odometry trajectories against GPS paths.
Running the Analysis
python3 analyze_iovnbd.py <path_to_v_dataset_csv>
Example:

python3 analyze_iovnbd.py "Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/V-S2.csv"