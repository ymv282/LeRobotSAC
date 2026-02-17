import mujoco
import mujoco.viewer
import os
import numpy as np
import time
model_path = os.path.expanduser("~/SO-ARM100/Simulation/SO101/scene.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)

    # Joint-Typ
    jtype = model.jnt_type[j]

    if jtype in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
        jmin, jmax = model.jnt_range[j]
        print(j, name, "min:", jmin, "max:", jmax)
    else:
        print(j, name, "keine Limits")
joint_min = np.array([
    -1.9198621771937616,
    -1.7453292519943224,
    -1.69,
    -1.6580628494556928,
    -2.7438472969992493,
    -0.17453297762778586
])

joint_max = np.array([
     1.9198621771937634,
     1.7453292519943366,
     1.69,
     1.6580627293335335,
     2.841206309382605,
     1.7453291995659765
])
def get_gripper_position():
        gripper_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
        )
        # Endeffektor Position
        return data.xpos[gripper_id].copy()
for i in range(1,10):
    for _ in range(10):
            mujoco.mj_step(model, data)
    print(f'Gripper Position: {get_gripper_position()}')
