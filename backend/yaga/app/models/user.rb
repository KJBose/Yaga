require 'mongo'
#Mongo::Logger.logger.level = ::Logger::INFO
#Mongo::Logger.logger.level = ::Logger::DEBUG

class User
#class User < ActiveRecord::Base
  include ActiveModel::Model	
  @@db = nil
  attr_accessor :id, :number, :first_name, :last_name, :gender, :usertype, :helps
  #@id=doc[:_id].to_s:_id=>BSON::ObjectId.from_string(@id)
  
  def initialize(params={})

	@id=params[:_id].nil? ? params[:id] : params[:_id].to_s
	@number=params[:number].to_i
	@first_name=params[:first_name]
	@last_name=params[:last_name]
	@gender=params[:gender]
	@usertype=params[:usertype]
	@helps=params[:helps]

  end

  #Implement a class method in the `Solution` class called `mongo_client` that will 
  def self.mongo_client
    #create a `Mongo::Client` connection to the server using a URL (.e.g., 'mongodb://localhost:27017')
    #configure the client to use the `test` database
    #assign the client to @@db instance variable and return that client
    
    #db = Mongo::Clients.new("mongodb://localhost:27017/test")
    client = Mongo::Client.new(['localhost:27017'], database: 'yaga')
    return client
  end

  #Implement a class method in the `Solution` class called `collection` that will
  def self.collection
    #return the `zips` collection
    return self.mongo_client['users']
    
  end

  #Implement CRUD apis Begin
  #def self.all(prototype={}, sort={:first_name=>1}, offset=0, limit=nil)
  def self.all(prototype={}, sort={:number=>1}, offset=0, limit=nil)
    #self.all
    tmp = {}
  	sort.each {|k,v|
    	k = k.to_sym==:number ? :number : k.to_sym
    	tmp[k] = v  if [:number, :first_name, :last_name, :gender, :usertype, :helps ].include?(k)
  	}
  	sort=tmp

  	prototype=prototype.symbolize_keys if !prototype.nil?

  	result=collection.find(prototype).projection({_id:true, number:true, first_name:true, last_name:true, gender:true, usertype:true, helps:true}).sort(sort).skip(offset)
  	result=result.limit(limit) if !limit.nil?

  	return result
    #return self.collection.find()
  end

  def self.find id
    Rails.logger.debug {"getting user #{id}"}

    doc = collection.find(:_id=>BSON::ObjectId.from_string(id))
                   .projection({_id:true, number:true, first_name:true, last_name:true, gender:true, usertype:true, helps:true})
                   .first

    
    return doc.nil? ? nil : User.new(doc)
  end

  #Implement an instance method in the `Solution` class called `sample` that will
  def sample
    #return a single document from the `zips` collection from the database. 
    #This does not have to be random. It can be first, last, or any other document in the collection.
    
    #return self.mongo_client['zips'].find.first
    
    return self.class.collection.find.first
  end

  def save
  	Rails.logger.debug {"saving #{"self"}"}
    bson_id = @id.present? ? BSON::ObjectId.from_string(@id) : BSON::ObjectId.new

  	self.class.collection.insert_one(
      _id: bson_id,
      number: @number,
      first_name: @first_name,
      last_name: @last_name,
      gender: @gender,
      usertype: @usertype,
      helps: @helps
    )

    @id = bson_id.to_s
    self
  end

  def update(params)
  	Rails.logger.debug {"updating #{"self"} with #{params}"}
  @number=params[:number].to_i
	@first_name=params[:first_name]
	@last_name=params[:last_name]
	@helps=params[:helps]
	@gender=params[:gender]
	@usertype=params[:usertype]

  updates = params.slice(:number, :first_name, :last_name, :gender, :usertype, :helps)
  updates[:number] = updates[:number].to_i if updates.key?(:number)

	return self.class.collection
			  .find(_id:BSON::ObjectId.from_string(@id))
			  .update_one('$set' => updates)
  end

  def destroy
	Rails.logger.debug {"destroying #{"self"}"}
  	return self.class.collection
			  .find(_id:BSON::ObjectId.from_string(@id))
			  .delete_one
  end 
  #Implement CRUD apis End

  #ActiveModel Framework Begin
  def persisted?

	return !@id.nil? ? true : false

  end

  def created_at

	nil

  end

  def updated_at

	nil

  end
  #ActiveModel Framework End

end

