(function () {
"use strict";

angular.module('common')
.service('HelpService', HelpService);


HelpService.$inject = ['$http', 'AuthApiPath'];
function HelpService($http, AuthApiPath) {
  var service = this;

  service.getPrioritizedHelps = function () {
    return $http.get(AuthApiPath + '/helps/prioritized').then(function (response) {
      return response.data.helps;
    });
  };

  service.getMyHelps = function () {
    return $http.get(AuthApiPath + '/helps/my').then(function (response) {
      return response.data.helps;
    });
  };

  service.getHelpsByProfile = function (profileId) {
    return $http.get(AuthApiPath + '/helps', {
      params: { profile_id: profileId }
    }).then(function (response) {
      return response.data.helps;
    });
  };

  service.createHelp = function (help) {
    return $http.post(AuthApiPath + '/helps', { help: help }).then(function (response) {
      return response.data.help;
    });
  };

  service.takeHelp = function (helpId) {
    return $http.post(AuthApiPath + '/helps/' + helpId + '/take').then(function (response) {
      return response.data.help;
    });
  };

  service.markInProgress = function (helpId) {
    return $http.post(AuthApiPath + '/helps/' + helpId + '/mark_in_progress').then(function (response) {
      return response.data.help;
    });
  };

  service.completeHelp = function (helpId) {
    return $http.post(AuthApiPath + '/helps/' + helpId + '/complete').then(function (response) {
      return response.data.help;
    });
  };

  service.cancelHelp = function (helpId) {
    return $http.post(AuthApiPath + '/helps/' + helpId + '/cancel').then(function (response) {
      return response.data.help;
    });
  };

  service.sendAppreciation = function (payload) {
    return $http.post(AuthApiPath + '/appreciations', { appreciation: payload }).then(function (response) {
      return response.data.appreciation;
    });
  };

  service.getHelpLedger = function (helpId) {
    return $http.get(AuthApiPath + '/helps/' + helpId + '/ledger_events').then(function (response) {
      return response.data.ledger_events || [];
    });
  };

  service.recordHelpLedgerEvent = function (helpId, payload) {
    return $http.post(AuthApiPath + '/helps/' + helpId + '/ledger_events', { ledger_event: payload }).then(function (response) {
      return response.data.ledger_event;
    });
  };

  service.updateHelpLedgerEvent = function (helpId, eventId, payload) {
    return $http.patch(AuthApiPath + '/helps/' + helpId + '/ledger_events/' + eventId, { ledger_event: payload }).then(function (response) {
      return response.data.ledger_event;
    });
  };
}

})();
