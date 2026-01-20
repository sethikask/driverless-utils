**-----On first installation-----**

*Requires python3.11*

mkdir python-client

cd python-client

git clone https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator

*-----Delete all but 'python' folder, and move contents into python-client-----*

cd ..

python -m venv fsds-env

./fsds-env/scripts/activate

pip install msgpack-rpc-python numpy opencv-contrib-python pandas matplotlib PyQt5



**-----On startup-----**

./fsds-env/scripts/activate



**-----Running the sim-----**

*Run FSDS.exe*

*Select 'CustomMap' and set the filepath to:*

\#C:\\MiscPrograms\\HFR\\fsds-v2.2.0-windows\\data\\track\_droneportNew.csv

C:\\MiscPrograms\\HFR\\fsds-v2.2.0-windows\\data\\outputs\\handlingTrack\\output\_fsds.csv



**-----Running the autonomous system------**

python fsds.py

