(function () {
"use strict";

angular.module('public')
.controller('MyInfoController', MyInfoController);

MyInfoController.$inject = ['$state', '$stateParams', '$q', 'userInfo', 'AuthService', 'UserDataService', 'SocialService'];
function MyInfoController($state, $stateParams, $q, userInfo, AuthService, UserDataService, SocialService) {
  var infoCtrl = this;
  infoCtrl.isAuthenticated = AuthService.isAuthenticated();
  infoCtrl.readOnlyProfile = !!$stateParams.profileId;
  infoCtrl.userexist = false;
  infoCtrl.userNotexist = true;
  infoCtrl.profileNotFound = false;
  infoCtrl.profileRestricted = false;
  infoCtrl.profileRestrictedMessage = '';
  infoCtrl.error = '';
  infoCtrl.success = '';
  infoCtrl.processing = false;
  infoCtrl.currentUserProfileId = null;
  infoCtrl.helpHistory = {
    given: [],
    taken: [],
    given_count: 0,
    taken_count: 0
  };
  infoCtrl.leaderboard = [];
  infoCtrl.currentRank = null;
  infoCtrl.networkHiddenMessage = '';
  infoCtrl.relationship = {
    viewer_follows_target: false,
    target_follows_viewer: false,
    can_follow: false,
    can_unfollow: false
  };
  infoCtrl.socialGraph = {
    followers: [],
    following: [],
    follower_count: 0,
    following_count: 0
  };
  infoCtrl.selectedInterests = SocialService.getSelectedInterests();

  if (!userInfo) {
    infoCtrl.userNotexist = true;
    infoCtrl.profileNotFound = infoCtrl.readOnlyProfile;
    return;
  }

  if (userInfo.restricted) {
    infoCtrl.userNotexist = true;
    infoCtrl.profileRestricted = true;
    infoCtrl.profileRestrictedMessage = userInfo.restricted_message || 'This profile is private based on user settings.';
    return;
  }

  if (!infoCtrl.readOnlyProfile && !infoCtrl.isAuthenticated) {
    infoCtrl.userNotexist = true;
    return;
  }

  infoCtrl.userNotexist = false;
  infoCtrl.userexist = true;
  infoCtrl.dashboard = userInfo;
  infoCtrl.profileForm = angular.copy(userInfo.profile || {});
  infoCtrl.profileForm.profile_visibility = infoCtrl.profileForm.profile_visibility || 'public';
  infoCtrl.profileForm.network_visibility = infoCtrl.profileForm.network_visibility || 'public';
  infoCtrl.initials = initialsFromName(infoCtrl.dashboard.profile && infoCtrl.dashboard.profile.full_name);
  infoCtrl.currentUserProfileId = (!infoCtrl.readOnlyProfile && infoCtrl.dashboard.profile && infoCtrl.dashboard.profile.id) || null;

  if (infoCtrl.isAuthenticated && infoCtrl.readOnlyProfile) {
    AuthService.getCurrentUser()
      .then(function (current) {
        infoCtrl.currentUserProfileId = current && current.profile && current.profile.id;
      })
      .catch(function () {
        infoCtrl.currentUserProfileId = null;
      });
  }

  infoCtrl.loadProfileInsights = function () {
    var profileId = infoCtrl.dashboard.profile && infoCtrl.dashboard.profile.id;
    if (!profileId) {
      return;
    }

    $q.all([
      SocialService.getUserHelpHistory(profileId),
      SocialService.getLeaderboard(),
      SocialService.getSocialGraph(profileId).catch(function (error) {
        if (error && error.status === 403) {
          return {
            followers: [],
            following: [],
            follower_count: 0,
            following_count: 0,
            can_follow: false,
            can_unfollow: false,
            viewer_follows_target: false,
            target_follows_viewer: false,
            network_hidden_message: (error.data && error.data.error) || 'This user has restricted their network visibility.'
          };
        }

        return SocialService.buildFallbackSocialGraph(infoCtrl.dashboard.profile, infoCtrl.leaderboard);
      })
    ]).then(function (results) {
      var history = results[0] || {};
      var leaderboardPayload = results[1] || {};
      var socialPayload = results[2] || {};

      infoCtrl.helpHistory = {
        given: history.given || [],
        taken: history.taken || [],
        given_count: history.given_count || 0,
        taken_count: history.taken_count || 0
      };

      infoCtrl.leaderboard = leaderboardPayload.leaderboard || [];
      infoCtrl.socialGraph = {
        followers: socialPayload.followers || [],
        following: socialPayload.following || [],
        follower_count: socialPayload.follower_count || 0,
        following_count: socialPayload.following_count || 0
      };
      infoCtrl.relationship = {
        viewer_follows_target: !!socialPayload.viewer_follows_target,
        target_follows_viewer: !!socialPayload.target_follows_viewer,
        can_follow: !!socialPayload.can_follow,
        can_unfollow: !!socialPayload.can_unfollow
      };
      infoCtrl.networkHiddenMessage = socialPayload.network_hidden_message || '';
      infoCtrl.currentRank = leaderboardPayload.current_user_rank;

      if (!infoCtrl.currentRank && profileId) {
        var row = infoCtrl.leaderboard.find(function (entry) {
          return entry.profile && entry.profile.id === profileId;
        });
        infoCtrl.currentRank = row && row.rank;
      }
    }).catch(function () {
      infoCtrl.error = 'Unable to load profile insights.';
    });
  };

  infoCtrl.updateProfile = function () {
    if (infoCtrl.readOnlyProfile) {
      return;
    }

    infoCtrl.error = '';
    infoCtrl.success = '';
    infoCtrl.processing = true;

    UserDataService.updateProfile(infoCtrl.profileForm)
      .then(function (response) {
        infoCtrl.dashboard.profile = response.profile;
        infoCtrl.dashboard.help_counts = response.help_counts;
        infoCtrl.dashboard.appreciation = response.appreciation;
        infoCtrl.initials = initialsFromName(infoCtrl.dashboard.profile && infoCtrl.dashboard.profile.full_name);
        infoCtrl.success = 'Profile updated.';
        infoCtrl.loadProfileInsights();
      })
      .catch(function (error) {
        var apiErrors = error && error.data && error.data.errors;
        infoCtrl.error = apiErrors ? apiErrors.join(', ') : 'Unable to update profile.';
      })
      .finally(function () {
        infoCtrl.processing = false;
      });
  };

  infoCtrl.signOut = function () {
    AuthService.signOut().finally(function () {
      $state.go('public.index');
    });
  };

  infoCtrl.canOpenProfile = function (profileId) {
    return !!profileId &&
      profileId.indexOf('dummy-') !== 0 &&
      profileId.indexOf('mock-') !== 0 &&
      profileId.indexOf('self-') !== 0;
  };

  infoCtrl.openProfile = function (profileId) {
    if (!infoCtrl.canOpenProfile(profileId)) {
      return;
    }

    if (infoCtrl.currentUserProfileId && profileId === infoCtrl.currentUserProfileId) {
      $state.go('public.profile');
      return;
    }

    $state.go('public.userProfile', { profileId: profileId });
  };

  infoCtrl.canStartChat = function (profileId) {
    return infoCtrl.isAuthenticated &&
      infoCtrl.canOpenProfile(profileId) &&
      (!infoCtrl.currentUserProfileId || profileId !== infoCtrl.currentUserProfileId);
  };

  infoCtrl.startChat = function (profileId) {
    if (!infoCtrl.canStartChat(profileId)) {
      return;
    }

    $state.go('public.chat', { withProfileId: profileId });
  };

  infoCtrl.canFollowTarget = function () {
    if (!infoCtrl.isAuthenticated || !infoCtrl.readOnlyProfile) {
      return false;
    }

    var profileId = infoCtrl.dashboard && infoCtrl.dashboard.profile && infoCtrl.dashboard.profile.id;
    if (!profileId || !infoCtrl.currentUserProfileId || profileId === infoCtrl.currentUserProfileId) {
      return false;
    }

    return !!infoCtrl.relationship.can_follow;
  };

  infoCtrl.canUnfollowTarget = function () {
    if (!infoCtrl.isAuthenticated || !infoCtrl.readOnlyProfile) {
      return false;
    }

    var profileId = infoCtrl.dashboard && infoCtrl.dashboard.profile && infoCtrl.dashboard.profile.id;
    if (!profileId || !infoCtrl.currentUserProfileId || profileId === infoCtrl.currentUserProfileId) {
      return false;
    }

    return !!infoCtrl.relationship.can_unfollow;
  };

  infoCtrl.followTarget = function () {
    var profileId = infoCtrl.dashboard && infoCtrl.dashboard.profile && infoCtrl.dashboard.profile.id;
    if (!profileId || !infoCtrl.canFollowTarget()) {
      return;
    }

    infoCtrl.error = '';
    SocialService.followProfile(profileId)
      .then(function () {
        infoCtrl.loadProfileInsights();
      })
      .catch(function (error) {
        infoCtrl.error = (error && error.data && (error.data.error || (error.data.errors && error.data.errors.join(', ')))) ||
          'Unable to follow this user.';
      });
  };

  infoCtrl.unfollowTarget = function () {
    var profileId = infoCtrl.dashboard && infoCtrl.dashboard.profile && infoCtrl.dashboard.profile.id;
    if (!profileId || !infoCtrl.canUnfollowTarget()) {
      return;
    }

    infoCtrl.error = '';
    SocialService.unfollowProfile(profileId)
      .then(function () {
        infoCtrl.loadProfileInsights();
      })
      .catch(function (error) {
        infoCtrl.error = (error && error.data && (error.data.error || (error.data.errors && error.data.errors.join(', ')))) ||
          'Unable to unfollow this user.';
      });
  };

  infoCtrl.loadProfileInsights();

  function initialsFromName(name) {
    var safe = (name || '').trim();
    if (!safe) {
      return 'YU';
    }

    var tokens = safe.split(/\s+/).filter(Boolean);
    if (tokens.length === 1) {
      return tokens[0].slice(0, 2).toUpperCase();
    }

    return (tokens[0][0] + tokens[1][0]).toUpperCase();
  }
}

})();
