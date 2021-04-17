from gym.envs.registration import register

register(
    id='TwoRooms-v0',
    entry_point='gym_two_rooms.envs:TwoRoomsEnv'
    )

register(
    id='TreasureMap-v0',
    entry_point='gym_two_rooms.envs:TreasureMapEnv'
)

register(
    id='TreasureMapHard-v0',
    entry_point='gym_two_rooms.envs:TreasureMapHardEnv'
)
