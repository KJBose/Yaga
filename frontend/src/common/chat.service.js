(function () {
"use strict";

angular.module('common')
.service('ChatService', ChatService);


ChatService.$inject = ['$http', 'AuthApiPath'];
function ChatService($http, AuthApiPath) {
  var service = this;

  service.getConversations = function () {
    return $http.get(AuthApiPath + '/chats').then(function (response) {
      return response.data.conversations || [];
    });
  };

  service.startConversation = function (participantProfileId) {
    return $http.post(AuthApiPath + '/chats', {
      participant_profile_id: participantProfileId
    }).then(function (response) {
      return response.data;
    });
  };

  service.getConversation = function (chatId, limit) {
    return $http.get(AuthApiPath + '/chats/' + chatId, {
      params: { limit: limit || 80 }
    }).then(function (response) {
      return response.data;
    });
  };

  service.getMessages = function (chatId, options) {
    var params = {};
    if (options && options.after) {
      params.after = options.after;
    }
    if (options && options.limit) {
      params.limit = options.limit;
    }

    return $http.get(AuthApiPath + '/chats/' + chatId + '/messages', {
      params: params
    }).then(function (response) {
      return response.data.messages || [];
    });
  };

  service.sendMessage = function (chatId, body) {
    return $http.post(AuthApiPath + '/chats/' + chatId + '/messages', {
      message: { body: body }
    }).then(function (response) {
      return response.data.message;
    });
  };
}

})();
