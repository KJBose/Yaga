(function () {
"use strict";

angular.module('public')
.controller('HomeController', HomeController);

HomeController.$inject = ['$state', '$q', 'AuthService', 'UserDataService', 'SocialService'];
function HomeController($state, $q, AuthService, UserDataService, SocialService) {
  var homeCtrl = this;

  homeCtrl.isAuthenticated = AuthService.isAuthenticated();
  homeCtrl.loadingFeed = false;
  homeCtrl.error = '';
  homeCtrl.summary = {
    requested: 0,
    given: 0,
    taken: 0
  };
  homeCtrl.updates = [];
  homeCtrl.feedView = 'all';
  homeCtrl.socialGraph = {
    followers: [],
    following: [],
    follower_count: 0,
    following_count: 0
  };
  homeCtrl.topLeaders = [];
  homeCtrl.interestOptions = SocialService.getInterestOptions();
  homeCtrl.selectedInterests = SocialService.getSelectedInterests();
  homeCtrl.currentProfileId = null;

  homeCtrl.isSelectedInterest = function (interestId) {
    return homeCtrl.selectedInterests.indexOf(interestId) >= 0;
  };

  homeCtrl.toggleInterest = function (interestId) {
    var index = homeCtrl.selectedInterests.indexOf(interestId);
    if (index >= 0) {
      homeCtrl.selectedInterests.splice(index, 1);
      return;
    }

    homeCtrl.selectedInterests.push(interestId);
  };

  homeCtrl.saveInterests = function () {
    SocialService.saveSelectedInterests(homeCtrl.selectedInterests);
    homeCtrl.refreshFeed();
  };

  homeCtrl.setFeedView = function (view) {
    homeCtrl.feedView = view;
  };

  homeCtrl.isFeedView = function (view) {
    return homeCtrl.feedView === view;
  };

  homeCtrl.visibleUpdates = function () {
    if (homeCtrl.feedView === 'all') {
      return homeCtrl.updates;
    }

    return (homeCtrl.updates || []).filter(function (update) {
      return update.update_type === homeCtrl.feedView;
    });
  };

  homeCtrl.openLogin = function () {
    $state.go('public.signin');
  };

  homeCtrl.openSupport = function () {
    $state.go('public.menu');
  };

  homeCtrl.canOpenProfile = function (profileId) {
    return !!profileId &&
      profileId.indexOf('dummy-') !== 0 &&
      profileId.indexOf('mock-') !== 0 &&
      profileId.indexOf('self-') !== 0;
  };

  homeCtrl.openUserProfile = function (profileId) {
    if (!homeCtrl.canOpenProfile(profileId)) {
      return;
    }

    if (homeCtrl.currentProfileId && profileId === homeCtrl.currentProfileId) {
      $state.go('public.profile');
      return;
    }

    $state.go('public.userProfile', { profileId: profileId });
  };

  homeCtrl.canStartChat = function (profileId) {
    return homeCtrl.isAuthenticated &&
      homeCtrl.canOpenProfile(profileId) &&
      (!homeCtrl.currentProfileId || profileId !== homeCtrl.currentProfileId);
  };

  homeCtrl.startChat = function (profileId) {
    if (!homeCtrl.canStartChat(profileId)) {
      return;
    }

    $state.go('public.chat', { withProfileId: profileId });
  };

  homeCtrl.refreshFeed = function () {
    if (!homeCtrl.isAuthenticated) {
      return;
    }

    homeCtrl.loadingFeed = true;
    homeCtrl.error = '';

    $q.all([
      UserDataService.getProfile(),
      SocialService.getLeaderboard()
    ]).then(function (result) {
      var profilePayload = result[0] || {};
      var leaderboardPayload = result[1] || {};
      var profile = profilePayload.profile;
      homeCtrl.currentProfileId = profile && profile.id;

      return SocialService.getSocialGraph(homeCtrl.currentProfileId).catch(function () {
        return SocialService.buildFallbackSocialGraph(profile, leaderboardPayload.leaderboard);
      }).then(function (graphPayload) {
        homeCtrl.socialGraph = graphPayload || {
          followers: [],
          following: [],
          follower_count: 0,
          following_count: 0
        };
        homeCtrl.topLeaders = (leaderboardPayload.leaderboard || []).slice(0, 6);

        return SocialService.getHomeFeed(profile, homeCtrl.selectedInterests, homeCtrl.socialGraph);
      })
        .then(function (feedPayload) {
          homeCtrl.summary = feedPayload.summary;
          homeCtrl.updates = feedPayload.updates;
        });
    }).catch(function () {
      homeCtrl.error = 'Unable to load personalized updates right now.';
    }).finally(function () {
      homeCtrl.loadingFeed = false;
    });
  };

  if (homeCtrl.isAuthenticated) {
    homeCtrl.refreshFeed();
  }
}

})();
