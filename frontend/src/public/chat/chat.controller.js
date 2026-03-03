(function () {
"use strict";

angular.module('public')
.controller('ChatController', ChatController);

ChatController.$inject = ['$scope', '$state', '$stateParams', '$interval', 'AuthService', 'ChatService'];
function ChatController($scope, $state, $stateParams, $interval, AuthService, ChatService) {
  var chatCtrl = this;
  var poller = null;

  chatCtrl.isAuthenticated = AuthService.isAuthenticated();
  chatCtrl.loading = false;
  chatCtrl.syncing = false;
  chatCtrl.sending = false;
  chatCtrl.error = '';
  chatCtrl.conversations = [];
  chatCtrl.activeConversation = null;
  chatCtrl.messages = [];
  chatCtrl.newMessage = '';
  chatCtrl.currentUserProfileId = null;

  chatCtrl.openLogin = function () {
    $state.go('public.signin');
  };

  chatCtrl.openCreateAccount = function () {
    $state.go('public.createAccount');
  };

  chatCtrl.refreshConversations = function () {
    if (!chatCtrl.isAuthenticated) {
      return;
    }

    chatCtrl.syncing = true;
    return ChatService.getConversations()
      .then(function (conversations) {
        chatCtrl.conversations = conversations;
      })
      .catch(function () {
        chatCtrl.error = 'Unable to load conversations right now.';
      })
      .finally(function () {
        chatCtrl.syncing = false;
      });
  };

  chatCtrl.selectConversation = function (chatId) {
    if (!chatId) {
      return;
    }

    chatCtrl.error = '';
    chatCtrl.loading = true;

    ChatService.getConversation(chatId)
      .then(function (payload) {
        setActiveConversation(payload.conversation, payload.messages);
      })
      .catch(function (error) {
        chatCtrl.error = extractError(error, 'Unable to open this conversation.');
      })
      .finally(function () {
        chatCtrl.loading = false;
      });
  };

  chatCtrl.reloadActiveConversation = function () {
    if (!chatCtrl.activeConversation || !chatCtrl.activeConversation.id) {
      return;
    }

    chatCtrl.selectConversation(chatCtrl.activeConversation.id);
  };

  chatCtrl.sendMessage = function () {
    var body = (chatCtrl.newMessage || '').trim();
    if (!chatCtrl.activeConversation || !chatCtrl.activeConversation.id || !body) {
      return;
    }

    chatCtrl.sending = true;
    chatCtrl.error = '';

    ChatService.sendMessage(chatCtrl.activeConversation.id, body)
      .then(function (message) {
        chatCtrl.newMessage = '';
        mergeMessages([message]);
        chatCtrl.activeConversation.last_message_at = message.created_at;
        chatCtrl.activeConversation.last_message_preview = message.body;
        upsertConversation(chatCtrl.activeConversation);
      })
      .catch(function (error) {
        chatCtrl.error = extractError(error, 'Unable to send message.');
      })
      .finally(function () {
        chatCtrl.sending = false;
      });
  };

  chatCtrl.canOpenProfile = function (profileId) {
    return !!profileId &&
      profileId.indexOf('dummy-') !== 0 &&
      profileId.indexOf('mock-') !== 0 &&
      profileId.indexOf('self-') !== 0;
  };

  chatCtrl.openParticipantProfile = function (profileId) {
    if (!chatCtrl.canOpenProfile(profileId)) {
      return;
    }

    if (chatCtrl.currentUserProfileId && profileId === chatCtrl.currentUserProfileId) {
      $state.go('public.profile');
      return;
    }

    $state.go('public.userProfile', { profileId: profileId });
  };

  chatCtrl.activeOtherProfileId = function () {
    var conversation = chatCtrl.activeConversation;
    if (!conversation || !conversation.other_participant) {
      return null;
    }

    return conversation.other_participant.id;
  };

  chatCtrl.conversationName = function (conversation) {
    var other = conversation && conversation.other_participant;
    if (other && other.full_name) {
      return other.full_name;
    }

    return 'Conversation';
  };

  chatCtrl.conversationHeadline = function (conversation) {
    var other = conversation && conversation.other_participant;
    if (!other) {
      return 'Yaga member';
    }

    if (other.organization_name) {
      return (other.user_type || 'Individual') + ' | ' + other.organization_name;
    }

    return (other.user_type || 'Individual') + ' member';
  };

  chatCtrl.conversationPreview = function (conversation) {
    return (conversation && conversation.last_message_preview) || 'No messages yet.';
  };

  initialize();

  function initialize() {
    if (!chatCtrl.isAuthenticated) {
      return;
    }

    chatCtrl.loading = true;
    AuthService.getCurrentUser()
      .then(function (payload) {
        chatCtrl.currentUserProfileId = payload && payload.profile && payload.profile.id;
        return chatCtrl.refreshConversations();
      })
      .then(function () {
        if ($stateParams.withProfileId) {
          return startConversationWithProfile($stateParams.withProfileId);
        }

        var targetChatId = $stateParams.chatId || (chatCtrl.conversations[0] && chatCtrl.conversations[0].id);
        if (targetChatId) {
          chatCtrl.selectConversation(targetChatId);
        }
      })
      .finally(function () {
        chatCtrl.loading = false;
      });

    poller = $interval(pollUpdates, 6000);
    $scope.$on('$destroy', function () {
      if (poller) {
        $interval.cancel(poller);
        poller = null;
      }
    });
  }

  function pollUpdates() {
    if (!chatCtrl.isAuthenticated || chatCtrl.loading || chatCtrl.syncing) {
      return;
    }

    chatCtrl.refreshConversations();
    if (!chatCtrl.activeConversation || !chatCtrl.activeConversation.id) {
      return;
    }

    var after = mostRecentMessageTime();
    ChatService.getMessages(chatCtrl.activeConversation.id, { after: after, limit: 120 })
      .then(function (newMessages) {
        mergeMessages(newMessages || []);
      });
  }

  function startConversationWithProfile(profileId) {
    if (!profileId || (chatCtrl.currentUserProfileId && profileId === chatCtrl.currentUserProfileId)) {
      return;
    }

    chatCtrl.loading = true;
    return ChatService.startConversation(profileId)
      .then(function (payload) {
        setActiveConversation(payload.conversation, payload.messages);
      })
      .catch(function (error) {
        chatCtrl.error = extractError(error, 'Unable to start a conversation with this user.');
      })
      .finally(function () {
        chatCtrl.loading = false;
      });
  }

  function setActiveConversation(conversation, messages) {
    if (!conversation || !conversation.id) {
      return;
    }

    chatCtrl.activeConversation = conversation;
    chatCtrl.messages = sortedByCreatedAt(messages || []);
    upsertConversation(conversation);
    normalizeRouteParams(conversation.id);
  }

  function mergeMessages(incomingMessages) {
    if (!incomingMessages || incomingMessages.length === 0) {
      return;
    }

    var byId = {};
    chatCtrl.messages.forEach(function (message) {
      byId[message.id] = message;
    });

    incomingMessages.forEach(function (message) {
      byId[message.id] = message;
    });

    var merged = [];
    Object.keys(byId).forEach(function (id) {
      merged.push(byId[id]);
    });

    chatCtrl.messages = sortedByCreatedAt(merged);
  }

  function sortedByCreatedAt(messages) {
    return (messages || []).slice().sort(function (left, right) {
      return new Date(left.created_at).getTime() - new Date(right.created_at).getTime();
    });
  }

  function mostRecentMessageTime() {
    if (!chatCtrl.messages || chatCtrl.messages.length === 0) {
      return null;
    }

    return chatCtrl.messages[chatCtrl.messages.length - 1].created_at;
  }

  function upsertConversation(conversation) {
    var index = -1;
    (chatCtrl.conversations || []).forEach(function (row, rowIndex) {
      if (row.id === conversation.id) {
        index = rowIndex;
      }
    });

    if (index >= 0) {
      chatCtrl.conversations[index] = conversation;
    } else {
      chatCtrl.conversations.push(conversation);
    }

    chatCtrl.conversations.sort(function (left, right) {
      return new Date(right.last_message_at || right.updated_at).getTime() -
        new Date(left.last_message_at || left.updated_at).getTime();
    });
  }

  function normalizeRouteParams(chatId) {
    $state.go('public.chat', {
      chatId: chatId,
      withProfileId: null
    }, {
      notify: false,
      location: 'replace',
      inherit: true
    });
  }

  function extractError(error, fallback) {
    return (error && error.data && (error.data.error || (error.data.errors && error.data.errors.join(', ')))) || fallback;
  }
}

})();
