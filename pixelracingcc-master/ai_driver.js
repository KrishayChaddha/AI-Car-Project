/* ============================================================================
   AI DRIVER - Q-LEARNING DODGING SYSTEM (FIXED)
   ============================================================================ */
(function() {
  var AIDriver = {
    /* ----------------------------------------------------------
       BASIC SETTINGS
       ---------------------------------------------------------- */
    enabled: true,
    toggleButton: null,
    resetButton: null,
    episodeStatDom: null,
    epsilonStatDom: null,

    // Q-learning core
    q: {},
    actions: [0, 1, 2],   // 0 = left, 1 = straight, 2 = right
    alpha: 0.2,           // Boosted learning rate for faster adaptation
    gamma: 0.95,
    epsilon: 1.0,
    minEpsilon: 0.05,
    epsilonDecay: 0.995,  // Reaches full exploitation around episode 600

    lastState: null,
    lastAction: null,
    lastDodgeCount: 0,
    stuckFrames: 0,       // Tracks if car is stuck off-road

    // Performance metrics
    dodgeCount: 0,
    collisionCount: 0,
    currentEpisodeDodges: 0,
    episodeDodges: [],
    episodeCount: 0,
    prevPosition: 0,
    frameCount: 0,
    eligibleCars: null,

    // Decision / detection constants
    DECISION_INTERVAL: 5,
    LANE_WIDTH: 0.3,        // NARROWED: Prevents a single car from blocking all 3 lane sensors
    LOOKAHEAD_SEGMENTS: 50, // Reduced slightly for better immediate reactions
    DANGER_SEGMENTS: 15,

    /* ----------------------------------------------------------
       INITIALIZATION / PERSISTENCE
       ---------------------------------------------------------- */
    load: function() {
      try {
        var saved = Dom.storage.ai_q;
        this.q = saved ? JSON.parse(saved) : {};
      } catch (err) {
        this.q = {};
      }
    },
    save: function() {
      try { Dom.storage.ai_q = JSON.stringify(this.q); } catch (err) {}
    },
    reset: function() {
      this.dodgeCount = 0;
      this.collisionCount = 0;
      this.currentEpisodeDodges = 0;
      this.episodeDodges = [];
      this.episodeCount = 0;
      this.prevPosition = 0;
      this.frameCount = 0;
      this.stuckFrames = 0;
      this.eligibleCars = new Map();
      this.lastState = null;
      this.lastAction = null;
      this.lastDodgeCount = 0;
      this.load();
    },

    /* ----------------------------------------------------------
       UI BINDING
       ---------------------------------------------------------- */
    bindUI: function() {
      var self = this;
      var onReady = function() {
        self.toggleButton = document.getElementById('ai_toggle');
        if (self.toggleButton) {
          self.toggleButton.addEventListener('click', function() {
            self.enabled = !self.enabled;
            self.toggleButton.textContent = self.enabled ? 'AI: ON' : 'AI: OFF';
            self.toggleButton.className = self.enabled ? 'ai-toggle on' : 'ai-toggle';
          });
        }
        self.resetButton = document.getElementById('ai_reset_btn');
        if (self.resetButton) {
          self.resetButton.addEventListener('click', function() {
            self.q = {};
            self.epsilon = 1.0;
            self.episodeDodges = [];
            self.episodeCount = 0;
            self.currentEpisodeDodges = 0;
            self.lastState = null;
            self.lastAction = null;
            self.save();
            if (window.updateEpisodeChart) window.updateEpisodeChart(self.episodeDodges);
            self.updateStatsHUD();
          });
        }
        self.episodeStatDom = document.getElementById('ai_episode_value');
        self.epsilonStatDom = document.getElementById('ai_epsilon_value');
        self.updateStatsHUD();
      };
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', onReady);
      } else { onReady(); }
    },
    updateStatsHUD: function() {
      if (this.episodeStatDom) this.episodeStatDom.textContent = this.episodeCount;
      if (this.epsilonStatDom) this.epsilonStatDom.textContent = this.epsilon.toFixed(2);
    },

    /* ----------------------------------------------------------
       SENSING HELPERS
       ---------------------------------------------------------- */
    getCarsAhead: function(maxDist) {
      var result = [];
      if (typeof cars === 'undefined') return result;
      for (var i = 0; i < cars.length; i++) {
        var relZ = cars[i].z - position;
        if (relZ > 0 && relZ < maxDist) {
          result.push({ car: cars[i], dist: relZ, offset: cars[i].offset });
        }
      }
      return result;
    },
    nearestCarInLane: function(targetX, maxDist) {
      var nearest = Infinity;
      var carsAhead = this.getCarsAhead(maxDist);
      for (var i = 0; i < carsAhead.length; i++) {
        if (Math.abs(carsAhead[i].offset - targetX) < this.LANE_WIDTH) {
          if (carsAhead[i].dist < nearest) nearest = carsAhead[i].dist;
        }
      }
      return nearest;
    },

    /* ----------------------------------------------------------
       Q-LEARNING: STATE / ACTION / REWARD
       ---------------------------------------------------------- */
    getStateKey: function() {
      var segment = typeof findSegment !== 'undefined' ? findSegment(position + playerZ) : { curve: 0 };
      var curve = segment.curve || 0;
      var lookaheadDist = (typeof segmentLength !== 'undefined' ? segmentLength : 200) * this.LOOKAHEAD_SEGMENTS;
      
      // Spaced targets ensure distinct lane sensing
      var centerGap = this.nearestCarInLane(playerX, lookaheadDist);
      var leftGap = this.nearestCarInLane(playerX - 0.7, lookaheadDist);
      var rightGap = this.nearestCarInLane(playerX + 0.7, lookaheadDist);

      // Compressed State Space: 5 offsets (instead of 7), 3 curves (instead of 5)
      var offsetBucket = Math.max(-2, Math.min(2, Math.round(playerX * 1.5)));
      var curveBucket = Math.max(-1, Math.min(1, Math.round(curve)));
      
      // 3 distance buckets: 0=Clear, 1=Medium, 2=Danger
      var bucketGap = function(g) {
        var ratio = g / lookaheadDist;
        return ratio > 0.6 ? 0 : ratio > 0.3 ? 1 : 2; 
      };

      return offsetBucket + '_' + curveBucket + '_' +
             bucketGap(centerGap) + '_' + bucketGap(leftGap) + '_' + bucketGap(rightGap);
    },
    getQ: function(state) {
      if (!this.q[state]) this.q[state] = { 0: 0, 1: 0, 2: 0 };
      return this.q[state];
    },
    chooseAction: function(state) {
      if (Math.random() < this.epsilon) return this.actions[Math.floor(Math.random() * this.actions.length)];
      var qvals = this.getQ(state);
      var best = this.actions[0], bestVal = -Infinity;
      for (var i = 0; i < this.actions.length; i++) {
        var a = this.actions[i];
        if (qvals[a] > bestVal) { bestVal = qvals[a]; best = a; }
      }
      return best;
    },
    updateQ: function(state, action, reward, nextState) {
      var qvals = this.getQ(state);
      var nextQ = this.getQ(nextState);
      var nextMax = Math.max(nextQ[0], nextQ[1], nextQ[2]);
      qvals[action] += this.alpha * (reward + this.gamma * nextMax - qvals[action]);
    },
    computeReward: function() {
      var reward = 0.1; 
      var spdRatio = (typeof speed !== 'undefined' && typeof maxSpeed !== 'undefined' && maxSpeed > 0) ? (speed / maxSpeed) : 0;
      reward += spdRatio * 0.4;
      
      // Only penalize if completely off the road (grass). Don't punish using the side lanes to dodge!
      if (Math.abs(playerX) > 1.0) reward -= 0.5; 

      var dangerDist = (typeof segmentLength !== 'undefined' ? segmentLength : 200) * this.DANGER_SEGMENTS;
      var gap = this.nearestCarInLane(playerX, dangerDist);
      if (gap < dangerDist) {
        var closeness = (dangerDist - gap) / dangerDist;
        reward -= 1.5 * closeness; // Scale penalty to prompt earlier lane changes
      }
      return reward;
    },

    /* ----------------------------------------------------------
       ACTION EXECUTION
       ---------------------------------------------------------- */
    applyAction: function(action) {
      if (typeof keyLeft !== 'undefined') {
        keyLeft = false;
        keyRight = false;
        // Never stop! Braking gets the AI permanently stuck. Force it to learn dodging.
        keyFaster = true; 
        keySlower = false;

        if (Math.abs(playerX) > 1.2) {
          keyLeft = playerX > 0;
          keyRight = playerX < 0;
        } else if (action === 0) {
          keyLeft = true;
        } else if (action === 2) {
          keyRight = true;
        }
      }
    },

    /* ----------------------------------------------------------
       DODGE COUNTING 
       ---------------------------------------------------------- */
    initEligibleCars: function() {
      var playerAbsZ = position + playerZ;
      this.eligibleCars = new Map();
      if (typeof cars !== 'undefined') {
        for (var i = 0; i < cars.length; i++) {
          this.eligibleCars.set(cars[i], cars[i].z > playerAbsZ);
        }
      }
    },
    updateDodgeCount: function() {
      if (typeof cars === 'undefined') return;
      if (!this.eligibleCars || this.eligibleCars.size === 0) this.initEligibleCars();
      var playerAbsZ = position + playerZ;
      for (var i = 0; i < cars.length; i++) {
        var car = cars[i];
        if (!this.eligibleCars.has(car)) this.eligibleCars.set(car, car.z > playerAbsZ);
        
        if (this.eligibleCars.get(car) === true && car.z < playerAbsZ) {
          this.eligibleCars.set(car, false);
          this.currentEpisodeDodges++;
        }
      }
      this.dodgeCount = this.currentEpisodeDodges;
    },
    endEpisode: function() {
      this.episodeDodges.push(this.currentEpisodeDodges);
      this.episodeCount++;
      this.currentEpisodeDodges = 0;
      this.eligibleCars = new Map();
      this.lastState = null;
      this.lastAction = null;
      this.lastDodgeCount = 0;
      this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
      this.save();
      this.updateStatsHUD();
      if (window.updateEpisodeChart) window.updateEpisodeChart(this.episodeDodges);
    },

    /* ----------------------------------------------------------
       MAIN UPDATE LOOP
       ---------------------------------------------------------- */
    update: function() {
      if (!this.enabled || typeof position === 'undefined') return;
      this.frameCount++;

      // STUCK DETECTION: If off-road speed drops to 0, you can't steer and episode never ends.
      if (typeof speed !== 'undefined' && speed < 50) {
        this.stuckFrames++;
      } else {
        this.stuckFrames = 0;
      }

      // Crash triggers on track-wrap OR being stuck stationary for ~1 second
      var crashed = (this.prevPosition > position + 100) || (this.stuckFrames > 40);

      this.updateDodgeCount();

      if (crashed) {
        if (this.lastState !== null) {
          this.updateQ(this.lastState, this.lastAction, -20, this.getStateKey());
        }
        this.collisionCount++;
        this.endEpisode();
        
        // RECOVERY: Force the car back onto the road so it can actually accelerate in the new episode
        if (typeof playerX !== 'undefined') playerX = 0;
        this.stuckFrames = 0;
      }

      this.prevPosition = position;

      if (this.frameCount % this.DECISION_INTERVAL !== 0) return;

      var state = this.getStateKey();
      if (this.lastState !== null && !crashed) {
        var dodgeDelta = this.currentEpisodeDodges - this.lastDodgeCount;
        var reward = this.computeReward() + dodgeDelta * 3.5; 
        this.updateQ(this.lastState, this.lastAction, reward, state);
      }

      var action = this.chooseAction(state);
      this.applyAction(action);

      this.lastState = state;
      this.lastAction = action;
      this.lastDodgeCount = this.currentEpisodeDodges;
    }
  };

  AIDriver.reset();
  AIDriver.bindUI();
  window.AIDriver = AIDriver;
})();