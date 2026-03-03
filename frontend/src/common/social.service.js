(function () {
"use strict";

angular.module('common')
.service('SocialService', SocialService);

SocialService.$inject = ['$http', '$q', '$window', 'AuthApiPath', 'HelpService'];
function SocialService($http, $q, $window, AuthApiPath, HelpService) {
  var service = this;
  var INTERESTS_KEY = 'yaga_interest_filters';
  var INTEREST_OPTIONS = [
    { id: 'health', label: 'Health' },
    { id: 'education', label: 'Education' },
    { id: 'environment', label: 'Environment' },
    { id: 'community', label: 'Community' },
    { id: 'food', label: 'Food Support' },
    { id: 'housing', label: 'Housing' },
    { id: 'elderly', label: 'Elder Care' },
    { id: 'youth', label: 'Youth' }
  ];

  service.getInterestOptions = function () {
    return angular.copy(INTEREST_OPTIONS);
  };

  service.getSelectedInterests = function () {
    var raw = $window.localStorage.getItem(INTERESTS_KEY);
    if (!raw) {
      return ['health', 'education', 'community'];
    }

    try {
      var parsed = JSON.parse(raw);
      if (!angular.isArray(parsed)) {
        return ['health', 'education', 'community'];
      }

      return parsed.map(normalizeInterest).filter(Boolean);
    } catch (e) {
      return ['health', 'education', 'community'];
    }
  };

  service.saveSelectedInterests = function (interestIds) {
    var normalized = (interestIds || []).map(normalizeInterest).filter(Boolean);
    $window.localStorage.setItem(INTERESTS_KEY, JSON.stringify(unique(normalized)));
  };

  service.getSocialGraph = function (profileId) {
    var params = {};
    if (profileId) {
      params.profile_id = profileId;
    }

    return $http.get(AuthApiPath + '/social/graph', { params: params }).then(function (response) {
      return response.data;
    });
  };

  service.buildFallbackSocialGraph = function (profile, leaderboardRows) {
    var networkProfiles = buildNetworkProfiles(leaderboardRows, profile && profile.id);
    if (networkProfiles.length > 0) {
      var followers = networkProfiles.slice(0, 4);
      var following = networkProfiles.slice(4, 8);

      if (following.length === 0) {
        following = networkProfiles.slice(0, 3);
      }

      return {
        followers: followers,
        following: following,
        follower_count: followers.length,
        following_count: following.length
      };
    }

    var fallbackFollowers = [
      { id: 'mock-f1', full_name: 'Aarav Mehta', headline: 'Community Nurse' },
      { id: 'mock-f2', full_name: 'Nina Park', headline: 'School Volunteer' },
      { id: 'mock-f3', full_name: 'Luis Ortega', headline: 'Urban Gardener' },
      { id: 'mock-f4', full_name: 'Maya Ibrahim', headline: 'Neighborhood Organizer' }
    ];
    var fallbackFollowing = [
      { id: 'mock-g1', full_name: 'Kofi Mensah', headline: 'Food Drive Coordinator' },
      { id: 'mock-g2', full_name: 'Emma Collins', headline: 'Youth Mentor' },
      { id: 'mock-g3', full_name: 'Ravi Nair', headline: 'Disaster Response Volunteer' }
    ];

    return {
      followers: fallbackFollowers,
      following: fallbackFollowing,
      follower_count: fallbackFollowers.length,
      following_count: fallbackFollowing.length
    };
  };

  service.followProfile = function (profileId) {
    return $http.post(AuthApiPath + '/social/follow', {
      profile_id: profileId
    }).then(function (response) {
      return response.data;
    });
  };

  service.unfollowProfile = function (profileId) {
    return $http.delete(AuthApiPath + '/social/follow/' + profileId).then(function (response) {
      return response.data;
    });
  };

  service.getLeaderboard = function () {
    return $http.get(AuthApiPath + '/leaderboard').then(function (response) {
      return response.data;
    }).catch(function () {
      return {
        current_user_rank: null,
        leaderboard: [
          dummyRank(1, 'Aarav Mehta', 72.4, 12, 7),
          dummyRank(2, 'Nina Park', 68.1, 10, 9),
          dummyRank(3, 'Luis Ortega', 63.0, 11, 6),
          dummyRank(4, 'Maya Ibrahim', 58.7, 8, 11),
          dummyRank(5, 'Emma Collins', 56.3, 9, 8)
        ]
      };
    });
  };

  service.getUserHelpHistory = function (profileId) {
    return fetchHelpsByProfile(profileId).then(function (helps) {
      var safeHelps = angular.isArray(helps) ? helps : [];
      var given = safeHelps.filter(function (help) {
        return help.helper && help.helper.id === profileId;
      });
      var taken = safeHelps.filter(function (help) {
        return help.requester && help.requester.id === profileId;
      });

      return {
        all: safeHelps,
        given: given,
        taken: taken,
        given_count: given.length,
        taken_count: taken.length
      };
    }).catch(function () {
      var dummy = dummyHelpHistory(profileId);
      return {
        all: dummy,
        given: dummy.filter(function (help) { return help.helper && help.helper.id === profileId; }),
        taken: dummy.filter(function (help) { return help.requester && help.requester.id === profileId; }),
        given_count: 2,
        taken_count: 2
      };
    });
  };

  service.getHomeFeed = function (profile, selectedInterests, socialGraph) {
    var normalizedInterests = unique((selectedInterests || []).map(normalizeInterest).filter(Boolean));
    return $q.all([
      HelpService.getPrioritizedHelps().catch(function () { return []; }),
      HelpService.getMyHelps().catch(function () { return []; })
    ]).then(function (result) {
      var globalHelps = angular.isArray(result[0]) ? result[0] : [];
      var myHelps = angular.isArray(result[1]) ? result[1] : [];
      var byId = {};
      var mergedHelps = [];

      globalHelps.concat(myHelps).forEach(function (help) {
        if (!help || !help.id || byId[help.id]) {
          return;
        }
        byId[help.id] = true;
        mergedHelps.push(help);
      });

      var feed = buildHelpEvents(mergedHelps, normalizedInterests, socialGraph, profile);

      feed.sort(function (a, b) {
        return new Date(b.happened_at).getTime() - new Date(a.happened_at).getTime();
      });

      if (feed.length < 10) {
        feed = feed.concat(buildNetworkMockEvents(normalizedInterests, socialGraph));
      }

      return {
        updates: feed.slice(0, 32),
        summary: summarizeFeed(feed)
      };
    });
  };

  function buildHelpEvents(helps, selectedInterests, socialGraph, profile) {
    var networkNames = collectNetworkNames(socialGraph);
    var profileId = profile && profile.id;

    return helps.filter(function (help) {
      var category = normalizeInterest(help && help.category);
      var interestMatch = selectedInterests.length === 0 || selectedInterests.indexOf(category) >= 0;
      var requesterName = ((help && help.requester && help.requester.full_name) || help.requester_label || '').toLowerCase();
      var helperName = ((help && help.helper && help.helper.full_name) || '').toLowerCase();
      var networkMatch = (requesterName && networkNames.indexOf(requesterName) >= 0) ||
        (helperName && networkNames.indexOf(helperName) >= 0);
      var fromCurrentUser = profileId && (
        (help.requester && help.requester.id === profileId) ||
        (help.helper && help.helper.id === profileId)
      );

      return interestMatch || networkMatch || fromCurrentUser;
    }).map(function (help) {
      var requesterName = (help.requester && help.requester.full_name) || help.requester_label || 'A community member';
      var helperName = (help.helper && help.helper.full_name) || '';
      var fromNetwork = networkNames.indexOf(requesterName.toLowerCase()) >= 0 ||
        (helperName && networkNames.indexOf(helperName.toLowerCase()) >= 0);
      var fromYou = profileId && (
        (help.requester && help.requester.id === profileId) ||
        (help.helper && help.helper.id === profileId)
      );
      var updateType = deriveUpdateType(help, profileId);
      if (!updateType) {
        return null;
      }

      return {
        id: 'help-' + help.id,
        source: fromYou ? 'you' : (fromNetwork ? 'network' : 'interest'),
        update_type: updateType,
        title: titleForUpdate(help, updateType),
        message: eventMessage(help, requesterName, helperName, updateType),
        category: (help.category || 'general'),
        happened_at: help.updated_at || help.created_at || new Date().toISOString(),
        status: help.status,
        help_type: help.help_type
      };
    }).filter(Boolean);
  }

  function buildNetworkMockEvents(selectedInterests, socialGraph) {
    var interests = selectedInterests.length > 0 ? selectedInterests : ['community', 'health'];
    var followers = (socialGraph && socialGraph.followers) || [];
    var following = (socialGraph && socialGraph.following) || [];
    var actors = followers.concat(following);

    if (actors.length === 0) {
      actors = [{ full_name: 'Yaga Community' }];
    }

    return actors.slice(0, 6).map(function (actor, idx) {
      var interest = interests[idx % interests.length];
      var now = new Date();
      now.setMinutes(now.getMinutes() - (idx + 1) * 11);

      return {
        id: 'mock-' + idx,
        source: 'network',
        update_type: idx % 3 === 0 ? 'requested' : (idx % 3 === 1 ? 'taken' : 'given'),
        title: actor.full_name + ' posted a ' + interest + ' update',
        message: idx % 3 === 0 ?
          (actor.full_name + ' requested support in ' + interest + '.') :
          (idx % 3 === 1 ?
            (actor.full_name + ' took a ' + interest + ' request.') :
            (actor.full_name + ' completed a ' + interest + ' contribution.')),
        category: interest,
        happened_at: now.toISOString(),
        status: 'open',
        help_type: idx % 2 === 0 ? 'request' : 'offer'
      };
    });
  }

  function eventMessage(help, requesterName, helperName, updateType) {
    if (updateType === 'given') {
      return (helperName || requesterName) + ' contributed help' + (requesterName ? ' for ' + requesterName : '') + '.';
    }
    if (updateType === 'taken') {
      return requesterName + ' had help taken up' + (helperName ? ' by ' + helperName : '') + '.';
    }
    return requesterName + ' requested help from the community.';
  }

  function titleForUpdate(help, updateType) {
    if (updateType === 'given') {
      return 'Help Given: ' + help.title;
    }
    if (updateType === 'taken') {
      return 'Help Taken: ' + help.title;
    }
    return 'Help Requested: ' + help.title;
  }

  function deriveUpdateType(help, profileId) {
    if (!help) {
      return null;
    }

    if (help.status === 'completed' && help.helper) {
      return 'given';
    }

    if ((help.status === 'matched' || help.status === 'in_progress') && help.helper) {
      return 'taken';
    }

    if (help.help_type === 'offer') {
      return 'given';
    }

    if (profileId && help.helper && help.helper.id === profileId) {
      return 'given';
    }

    if (profileId && help.requester && help.requester.id === profileId && help.helper) {
      return 'taken';
    }

    return 'requested';
  }

  function collectNetworkNames(socialGraph) {
    var names = [];
    var followers = (socialGraph && socialGraph.followers) || [];
    var following = (socialGraph && socialGraph.following) || [];
    followers.concat(following).forEach(function (person) {
      if (person && person.full_name) {
        names.push(person.full_name.toLowerCase());
      }
    });
    return names;
  }

  function summarizeFeed(feed) {
    var summary = {
      requested: 0,
      given: 0,
      taken: 0
    };

    (feed || []).forEach(function (event) {
      var key = event && event.update_type;
      if (key && summary[key] !== undefined) {
        summary[key] += 1;
      }
    });

    return summary;
  }

  function fetchHelpsByProfile(profileId) {
    if (HelpService && angular.isFunction(HelpService.getHelpsByProfile)) {
      return HelpService.getHelpsByProfile(profileId);
    }

    return $http.get(AuthApiPath + '/helps', {
      params: { profile_id: profileId }
    }).then(function (response) {
      return response.data.helps;
    });
  }

  function normalizeInterest(value) {
    return (value || '').toString().trim().toLowerCase();
  }

  function unique(values) {
    var seen = {};
    var result = [];
    values.forEach(function (value) {
      if (!value || seen[value]) {
        return;
      }
      seen[value] = true;
      result.push(value);
    });
    return result;
  }

  function dummyRank(rank, fullName, score, given, taken) {
    return {
      rank: rank,
      profile: {
        id: 'dummy-' + rank,
        full_name: fullName,
        user_type: 'Individual'
      },
      score: score,
      helps_given: given,
      helps_taken: taken,
      helps_contributed: given + taken,
      appreciation_points: Math.round(score * 2)
    };
  }

  function buildNetworkProfiles(leaderboardRows, currentProfileId) {
    var rows = angular.isArray(leaderboardRows) ? leaderboardRows : [];
    var seen = {};
    var profiles = [];

    rows.forEach(function (row) {
      var profile = row && row.profile ? row.profile : row;
      var profileId = profile && profile.id;
      if (!profile || !profileId || seen[profileId] || profileId === currentProfileId) {
        return;
      }

      seen[profileId] = true;
      profiles.push({
        id: profileId,
        full_name: profile.full_name || 'Community Member',
        headline: headlineForProfile(profile)
      });
    });

    return profiles;
  }

  function headlineForProfile(profile) {
    if (!profile) {
      return 'Yaga Member';
    }

    var userType = profile.user_type || 'Individual';
    if (profile.organization_name) {
      return userType + ' | ' + profile.organization_name;
    }

    return userType + ' Contributor';
  }

  function dummyHelpHistory(profileId) {
    var now = new Date();
    return [
      {
        id: 'd-help-1',
        title: 'Community pantry support',
        category: 'community',
        status: 'completed',
        help_type: 'offer',
        requester: { id: 'dummy-r1', full_name: 'Nina Park' },
        helper: { id: profileId, full_name: 'You' },
        updated_at: new Date(now.getTime() - 4 * 3600 * 1000).toISOString()
      },
      {
        id: 'd-help-2',
        title: 'Senior medicine pickup',
        category: 'health',
        status: 'in_progress',
        help_type: 'request',
        requester: { id: profileId, full_name: 'You' },
        helper: { id: 'dummy-h2', full_name: 'Aarav Mehta' },
        updated_at: new Date(now.getTime() - 9 * 3600 * 1000).toISOString()
      },
      {
        id: 'd-help-3',
        title: 'After-school mentoring',
        category: 'education',
        status: 'matched',
        help_type: 'offer',
        requester: { id: 'dummy-r3', full_name: 'Luis Ortega' },
        helper: { id: profileId, full_name: 'You' },
        updated_at: new Date(now.getTime() - 21 * 3600 * 1000).toISOString()
      },
      {
        id: 'd-help-4',
        title: 'Neighborhood clean-up tools',
        category: 'environment',
        status: 'open',
        help_type: 'request',
        requester: { id: profileId, full_name: 'You' },
        helper: null,
        updated_at: new Date(now.getTime() - 27 * 3600 * 1000).toISOString()
      }
    ];
  }
}

})();
