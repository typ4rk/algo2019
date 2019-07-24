# algo2019 

- 개별 브랜치에서 작업하여 강화학습 테스트를 합니다.
- 개별 브랜치를 업로드 하여 변경사항을 공유합니다.
- 학습이 성공적(!) 으로 이루어진 경우 (또는 공통 반영이 필요한 경우) master 브랜치에 머지 해 주세요.

## Todo

- **dqn_custom_client.py 구현**
   - compute_reward(): 보상함수 작성 
      - Cornering 고려
      - Avoiding collision 고려
      - dqn_reward_tester.py run() 함수에서 테스트
   - action_space(): 액션 개수 지정
      - 액션을 세분화 시 학습이 수렴하지 않을 수 있음 (네트워크 노드와 레이어 수 변경 필요)
   - build_custom_model(): 네트워크 노드수 & 레이어 수 지정
      - 모델링 정보 변경시 새로 학습이 필요
   - make_dqn_param(): DQN Hyper-Parameters 변경
      - discount_factor = 0.99 (감가율)
      - learning_rate = 0.00025
      - epsilon = 1.0
      - epsilon_decay = 0.999
      - epsilon_min = 0.01
      - batch_size = 100 (Replay memory 에서 랜덤 추출하는 데이터 개수)
      - train_start =1000 (Replay memory 에서 학습 시작하는 데이터 개수)
      - memory_size = 20000 (Replay memory 사이즈)
   
- **airsim_env.py 구현**
   - get_current_state()
      - way_points(10 미터) 기준 에이전트가 관찰하는 정보 6가지를 반환  
         - forward_angle: 차량 전방의 도로가 휘어진 각도 (2번째 인덱스 값 반환됨)
         - moving_angle: 주행 시 차량의 운행 각
         - dist: 도로 중앙 차선으로부터 차량까지 직선거리(m)
         - velocity: 현재 차량 속도
         - o_to_middle: 도로 중앙 차선으로부터 장애물까지 직선거리(m)
         - o_dist: 차량에서 장애물까지 남은 거리(m)
   
## Tutorial

-   [공통가이드(영상)](https://www.youtube.com/playlist?list=PLF__5qnRIf5oOh7ejDMPt6bt1IPOQDWYG)
-   [7/10 Meetup(영상)](https://www.youtube.com/watch?v=UXeuZhcGzfI&feature=youtu.be)
-   [Guide,QuickStart(상세설명)](https://github.com/namojo/Algo2019)
-   [소스코드 다운로드](https://drive.google.com/open?id=1fkf2ihHAxDxyMN9ABvPnUzwMGZhUcGJm)


## Reference 

-   [파이썬 딥러닝](https://m.blog.naver.com/ssdyka/221299637545)
-   [How to Simulate a Self-Driving Car(영상)](https://www.youtube.com/watch?v=EaY5QiZwSP4)

## 대회일정
- 친선경기: 7/5 ~ 8/7
- 본선경기: 8/8 ~ 8/22 (2 Tracks)
